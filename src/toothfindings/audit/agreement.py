"""Inter-observer agreement on the double-read crops.

For the seven-category scheme and two collapsed views (four clinical classes; finding vs
sound): raw agreement, Cohen's kappa, PABAK (prevalence- and bias-adjusted kappa), Gwet's AC1
and positive specific agreement, with 2,000-resample bootstrap CIs clustered by radiograph
(``agreement.json``). Also Byrt's prevalence and bias indices and Krippendorff's alpha over
every reviewed crop (``agreement_extra.json``).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from ..utils import write_json
from .core import ALL7, CLIN, AuditData, percentile_ci


def coefficients(pairs: list[tuple[str, str]], cats: list[str]) -> dict:
    """Compute agreement coefficients of label pairs over the categories ``cats``.

    Parameters
    ----------
    pairs : list of (str, str)
        ``(reader 1 label, reader 2 label)`` per crop.
    cats : list of str
        Categories; their number ``K`` enters PABAK and AC1.

    Returns
    -------
    dict
        ``n``, ``raw_agreement``, ``cohen_kappa``, ``pabak``, ``gwet_ac1``,
        ``positive_specific_agreement`` per category and each reader's marginal counts;
        empty when there are no pairs.
    """
    n = len(pairs)
    if n == 0:
        return {}
    # pa: observed agreement; pe_*: chance agreement under each coefficient's model.
    pa = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe_cohen = sum((ca[k] / n) * (cb[k] / n) for k in cats)
    K = len(cats)
    pi = {k: (ca[k] / n + cb[k] / n) / 2 for k in cats}
    pe_ac1 = sum(pi[k] * (1 - pi[k]) for k in cats) / (K - 1)
    out = {
        "n": n,
        "raw_agreement": pa,
        "cohen_kappa": (pa - pe_cohen) / (1 - pe_cohen) if pe_cohen < 1 else float("nan"),
        "pabak": (K * pa - 1) / (K - 1),
        "gwet_ac1": (pa - pe_ac1) / (1 - pe_ac1) if pe_ac1 < 1 else float("nan"),
    }
    psa = {}
    for k in cats:
        both = sum(a == k and b == k for a, b in pairs)
        tot = ca[k] + cb[k]
        psa[k] = (2 * both / tot) if tot else None
    out["positive_specific_agreement"] = psa
    out["marginals_reader1"] = {k: ca[k] for k in cats}
    out["marginals_reader2"] = {k: cb[k] for k in cats}
    return out


def boot_ci(items, keyfn, statfn, n_boot: int = 2000, seed: int = 0) -> dict:
    """Compute cluster-bootstrap 95% intervals of the numeric statistics of ``statfn``.

    Parameters
    ----------
    items : list
        Items to resample.
    keyfn : callable
        Cluster of an item (the radiograph); clusters are drawn with replacement.
    statfn : callable
        Maps a resample to a dict of statistics; non-numeric and NaN values are skipped.
    n_boot : int
        Number of resamples.
    seed : int
        Seed of the ``random.Random`` generator.

    Returns
    -------
    dict
        ``{statistic: [low, high]}``.
    """
    rng = random.Random(seed)
    groups: dict = defaultdict(list)
    for it in items:
        groups[keyfn(it)].append(it)
    keys = list(groups)
    vals: dict = defaultdict(list)
    for _ in range(n_boot):
        sample: list = []
        for _ in range(len(keys)):
            sample += groups[keys[rng.randrange(len(keys))]]
        for k, v in statfn(sample).items():
            if isinstance(v, (int, float)) and v == v:  # v == v is False for NaN
                vals[k].append(v)
    return {k: percentile_ci(v) for k, v in vals.items()}


def _sound_finding(label: str) -> str:
    return "Sound" if label == "Healthy" else "Finding"


def agreement(data: AuditData, write: bool = True) -> dict:
    """Compute the agreement between the two readers of each double-read crop (``agreement.json``).

    Returns
    -------
    dict
        Per dataset: coefficients with ``ci95`` for the views ``all7``, ``clinical4`` (both
        labels clinical) and ``finding_vs_sound``; ``confusion_readers`` (``"label1|label2"``
        counts) and the model's mean confidence when the readers agree or disagree.
    """
    double_read = [r for r in data.wide() if int(r["n_readers"]) == 2]
    res = {}
    for ds in sorted({r["dataset"] for r in double_read}):
        sub = [r for r in double_read if r["dataset"] == ds]
        clin = [r for r in sub if r["label1"] in CLIN and r["label2"] in CLIN]
        views = {
            "all7": (ALL7, sub, lambda r: (r["label1"], r["label2"])),
            "clinical4": (CLIN, clin, lambda r: (r["label1"], r["label2"])),
            "finding_vs_sound": (
                ["Sound", "Finding"],
                clin,
                lambda r: (_sound_finding(r["label1"]), _sound_finding(r["label2"])),
            ),
        }
        res[ds] = {}
        for name, (cats, items, pair) in views.items():
            stats = coefficients([pair(r) for r in items], cats)
            pair_of = {r["blind_id"]: pair(r) for r in items}
            ci = boot_ci(
                items,
                lambda r: r["pan_id"],
                lambda s, pair_of=pair_of, cats=cats: coefficients([pair_of[r["blind_id"]] for r in s], cats),
            )
            stats["ci95"] = {k: ci[k] for k in ("raw_agreement", "cohen_kappa", "pabak", "gwet_ac1") if k in ci}
            res[ds][name] = stats
        cm = Counter((r["label1"], r["label2"]) for r in sub)
        res[ds]["confusion_readers"] = {f"{a}|{b}": n for (a, b), n in sorted(cm.items(), key=lambda x: -x[1])}
        agree = [float(r["pred_prob"]) for r in sub if r["label1"] == r["label2"]]
        dis = [float(r["pred_prob"]) for r in sub if r["label1"] != r["label2"]]
        res[ds]["model_confidence"] = {
            "mean_prob_when_readers_agree": float(np.mean(agree)) if agree else None,
            "mean_prob_when_readers_disagree": float(np.mean(dis)) if dis else None,
            "n_agree": len(agree),
            "n_disagree": len(dis),
        }
    if write:
        write_json(data.paths.review / "agreement.json", res)
    return res


def prevalence_bias(pairs, cats) -> dict:
    """Compute Byrt's prevalence and bias indices, one category against the rest.

    With ``a`` = both readers chose the category, ``d`` = neither, ``b``/``c`` = only reader 1 /
    only reader 2: prevalence index ``|a - d| / n`` and bias index ``|b - c| / n``.

    Returns
    -------
    dict
        ``{category: {"prevalence_index", "bias_index"}}``.
    """
    n = len(pairs)
    out = {}
    for k in cats:
        a = sum(1 for x, y in pairs if x == k and y == k)
        d = sum(1 for x, y in pairs if x != k and y != k)
        b = sum(1 for x, y in pairs if x == k and y != k)
        c = sum(1 for x, y in pairs if x != k and y == k)
        out[k] = {"prevalence_index": abs(a - d) / n, "bias_index": abs(b - c) / n}
    return out


def krippendorff_alpha(units: list[list[str]]) -> float | None:
    """Compute Krippendorff's alpha for nominal labels.

    Parameters
    ----------
    units : list of list of str
        Labels given to each unit (crop). Units read once enter only the expected disagreement.

    Returns
    -------
    float or None
        ``1 - Do / De`` (observed over expected disagreement); ``None`` when no unit has two
        labels or the expected disagreement is zero.
    """
    pairable = [u for u in units if len(u) >= 2]
    n_pairs = sum(len(u) * (len(u) - 1) for u in pairable)
    if n_pairs == 0:
        return None
    Do = sum(1 for u in pairable for i, a in enumerate(u) for j, b in enumerate(u) if i != j and a != b) / n_pairs
    vals = [v for u in units for v in u]
    n = len(vals)
    De = 1 - sum(c * (c - 1) for c in Counter(vals).values()) / (n * (n - 1))
    return 1 - Do / De if De else None


def agreement_extra(data: AuditData, write: bool = True) -> dict:
    """Compute the supplementary agreement statistics (``agreement_extra.json``).

    Returns
    -------
    dict
        Per dataset: crop counts, Byrt's indices over the seven categories (with the category
        of highest prevalence index and of largest bias index), Krippendorff's alpha over all
        crops and over double-read crops only, and label counts per reader.
    """
    res = {}
    for ds in ("tufts", "dentex"):
        sub = [r for r in data.wide() if r["dataset"] == ds]
        double_read = [r for r in sub if int(r["n_readers"]) == 2]
        entry: dict = {"n_all": len(sub), "n_double": len(double_read)}
        if double_read:
            pb = prevalence_bias([(r["label1"], r["label2"]) for r in double_read], ALL7)
            entry["prevalence_bias_all7"] = pb
            worst = max(pb.items(), key=lambda kv: kv[1]["prevalence_index"])
            entry["highest_prevalence_index"] = {"category": worst[0], **worst[1]}
            entry["largest_bias_index"] = max(((k, v["bias_index"]) for k, v in pb.items()), key=lambda kv: kv[1])
        units = [[r[k] for k in ("label1", "label2") if r.get(k)] for r in sub]
        entry["krippendorff_alpha_all_units"] = krippendorff_alpha(units)
        entry["krippendorff_alpha_double_only"] = krippendorff_alpha([u for u in units if len(u) == 2])
        labels_by_reader: dict = defaultdict(Counter)
        for r in sub:
            for k, lk in (("reader1", "label1"), ("reader2", "label2")):
                if r.get(k):
                    labels_by_reader[r[k]][r[lk]] += 1
        entry["reader_marginals"] = {k: dict(v) for k, v in labels_by_reader.items()}
        res[ds] = entry
    if write:
        write_json(data.paths.review / "agreement_extra.json", res)
    return res
