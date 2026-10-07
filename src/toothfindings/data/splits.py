"""Leakage-safe partitions.

* :func:`make_split`: the canonical four-class partition (``results/splits/splits``), patient
  level and global, a balanced training pool of 180 crops per class, every crop of the other
  patients as the fixed test set, and five ``StratifiedGroupKFold`` folds on the pool.
* :func:`make_splits_clean`: the contamination-free partitions (``splits_clean4``,
  ``splits_clean5``, ``splits_lodo_*``) built on ``manifest_clean.csv``. ``lodo`` stands for
  leave-one-dataset-out: training on some sources and testing on another.

.. warning::
   :func:`make_split` iterates Python ``set`` objects of patient ids (see
   ``docs/REPRODUCE.md``), so a re-run does not reproduce the saved training pool and folds.
   The saved files under ``results/splits/`` are the source of truth; the function is kept,
   unchanged, as documentation of how they were produced.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from ..utils import read_csv, write_csv, write_json

CLASSES = ["Implant", "Impacted", "Endodontics", "Healthy"]
CLASSES4 = ["Implant", "Impacted", "Endodontics", "Healthy"]
SPLIT_HEADER = ["filepath", "class", "source", "patient_id"]


def _rows4(data: list[dict]) -> list[list[str]]:
    """Return the :data:`SPLIT_HEADER` columns of each row as a list."""
    return [[r["filepath"], r["class"], r["source"], r["patient_id"]] for r in data]


def greedy_patient_assignment(
    by_patient: dict[str, list[dict]], patients: list[str], n_train: int
) -> tuple[set, set, Counter]:
    """Assign patients greedily: a patient goes to training while it helps fill any class below ``n_train``.

    Parameters
    ----------
    by_patient : dict
        Patient id -> crop rows.
    patients : list of str
        Patients in the (shuffled) visiting order.
    n_train : int
        Target number of training crops per class.

    Returns
    -------
    train_patients, test_patients : set
        Disjoint patient sets.
    train_counts : Counter
        Crops per class available to the training pool.
    """
    train_counts: Counter = Counter()
    train_patients, test_patients = set(), set()
    for p in patients:
        pclasses = Counter(r["class"] for r in by_patient[p])
        if any(train_counts[c] < n_train for c in pclasses):
            train_patients.add(p)
            train_counts += pclasses
        else:
            test_patients.add(p)
    return train_patients, test_patients, train_counts


def group_folds(
    pool: list[dict], group_key: str, n_folds: int = 5, seed: int = 0, as_int: bool = False
) -> dict[str, dict[str, list[int]]]:
    """Split a training pool into stratified, group-disjoint cross-validation folds.

    Parameters
    ----------
    pool : list of dict
        Rows with a ``class`` column.
    group_key : str
        Column holding the group (patient or radiograph).
    n_folds, seed : int
        ``StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)``.
    as_int : bool
        Store indices as Python ints (``list(map(int, ...))``) instead of ``ndarray.tolist()``.

    Returns
    -------
    dict
        ``{"0": {"train": [...], "val": [...]}, ...}`` with row indices into ``pool``.
    """
    y = [r["class"] for r in pool]
    groups = [r[group_key] for r in pool]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = {}
    for f, (tr, va) in enumerate(sgkf.split(pool, y, groups)):
        if {groups[i] for i in tr} & {groups[i] for i in va}:
            raise ValueError(f"patient leakage between training and validation in fold {f}")
        if as_int:
            folds[str(f)] = {"train": [int(i) for i in tr], "val": [int(i) for i in va]}
        else:
            folds[str(f)] = {"train": tr.tolist(), "val": va.tolist()}
    return folds


def write_fold_csvs(out: Path, pool: list[dict], folds: dict) -> None:
    """Write the explicit ``folds/fold<k>_{train,val}.csv`` image lists of each fold."""
    for f, idx in folds.items():
        for role in ("train", "val"):
            write_csv(Path(out) / "folds" / f"fold{f}_{role}.csv", _rows4([pool[i] for i in idx[role]]), SPLIT_HEADER)


def make_split(manifest: Path, out: Path, n_train: int = 180, n_folds: int = 5, seed: int = 0) -> dict:
    """Build the canonical patient-level partition from the crop manifest.

    Parameters
    ----------
    manifest : Path
        ``crops_source/manifest.csv``.
    out : Path
        Output folder (``test.csv``, ``train_pool.csv``, ``folds.json``, ``folds/``,
        ``split_manifest.csv``, ``split_summary.json``).
    n_train : int
        Balanced crops per class in the training pool.
    n_folds, seed : int
        Cross-validation folds and random seed.

    Returns
    -------
    dict
        The split summary.

    Notes
    -----
    Not reproducible across interpreter runs (set iteration order); see the module docstring.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = read_csv(manifest)

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)
    patients = list(by_patient.keys())
    rng.shuffle(patients)

    train_patients, test_patients, train_counts = greedy_patient_assignment(by_patient, patients, n_train)
    for c in CLASSES:
        if train_counts[c] < n_train:
            raise SystemExit(f"ERROR: class {c} has only {train_counts[c]} train-pool crops (< {n_train})")

    # Known issue: iterating a set makes the pool order depend on the string hash seed
    # (see docs/REPRODUCE.md).
    train_pool_by_class: dict[str, list[dict]] = defaultdict(list)
    for p in train_patients:
        for r in by_patient[p]:
            train_pool_by_class[r["class"]].append(r)
    train_pool = []
    for c in CLASSES:
        crops = train_pool_by_class[c][:]
        rng.shuffle(crops)
        train_pool.extend(crops[:n_train])
    rng.shuffle(train_pool)

    test_set = []
    for p in test_patients:
        test_set.extend(by_patient[p])

    tp = {r["patient_id"] for r in train_pool}
    sp = {r["patient_id"] for r in test_set}
    if tp & sp:
        raise ValueError(f"patient leakage: {len(tp & sp)} patients in both train and test")

    write_csv(out / "test.csv", _rows4(test_set), SPLIT_HEADER)
    write_csv(out / "train_pool.csv", _rows4(train_pool), SPLIT_HEADER)
    folds = group_folds(train_pool, "patient_id", n_folds, seed)
    write_json(out / "folds.json", folds)
    write_fold_csvs(out, train_pool, folds)
    write_csv(
        out / "split_manifest.csv",
        [[*r, "train_pool"] for r in _rows4(train_pool)] + [[*r, "test"] for r in _rows4(test_set)],
        [*SPLIT_HEADER, "partition"],
    )

    summary = {
        "n_train_per_class": n_train,
        "folds": n_folds,
        "seed": seed,
        "train_pool": dict(Counter(r["class"] for r in train_pool)),
        "test": dict(Counter(r["class"] for r in test_set)),
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "total_test_crops": len(test_set),
        "fold_val_sizes": {f: len(v["val"]) for f, v in folds.items()},
    }
    write_json(out / "split_summary.json", summary)
    return summary


def lyria_partition(rows: list[dict], n_train: int, seed: int, published: Path | None) -> tuple[set, set]:
    """Assign each primary-source patient to the training or test side of the clean partitions.

    Keeps the published partition (``split_manifest.csv``): its test and training patients
    keep their side. Patients it never saw are assigned with ``Random(seed)``, one in three to
    training. Falls back to the greedy rule of :func:`make_split` when the file is absent.

    Parameters
    ----------
    rows : list of dict
        Rows of ``manifest_clean.csv`` (``group`` = ``lyria:<pid>``).
    n_train, seed : int
        Greedy-rule target and seed.
    published : Path or None
        The published ``split_manifest.csv``.

    Returns
    -------
    train_groups, test_groups : set
        Disjoint ``lyria:<pid>`` groups, restricted to the groups present in ``rows``.
    """
    all_groups = {r["group"] for r in rows if r["source"] == "lyria"}
    if published is not None and Path(published).exists():
        tr, te = set(), set()
        for r in read_csv(published):
            if not r["filepath"].endswith(".png") or "lyria__" not in r["filepath"]:
                continue
            (tr if r["partition"] == "train_pool" else te).add(f"lyria:{r['patient_id']}")
    else:
        four_class_rows = [r for r in rows if r["source"] == "lyria" and r["class"] in CLASSES4]
        by_group: dict[str, list[dict]] = defaultdict(list)
        for r in four_class_rows:
            by_group[r["group"]].append(r)
        groups = list(by_group)
        random.Random(seed).shuffle(groups)
        tr, te, _ = greedy_patient_assignment(by_group, groups, n_train)
    # Patients absent from the published partition: one in three to training.
    rest = sorted(all_groups - tr - te)
    random.Random(seed).shuffle(rest)
    for i, g in enumerate(rest):
        (tr if i % 3 == 0 else te).add(g)
    return tr & all_groups, te & all_groups


def make_splits_clean(
    manifest: Path,
    out: Path,
    published: Path | None,
    train_sources: str = "lyria",
    test_sources: str = "lyria",
    classes: str = "Endodontics,Healthy,Impacted,Implant,Restored",
    cap: int = 180,
    class_caps: str = "",
    fill_class: str = "",
    fill_sources: str = "",
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Build an in-domain or leave-one-dataset-out (LODO) partition of the clean pool.

    Parameters
    ----------
    manifest : Path
        ``crops_source/manifest_clean.csv``.
    out : Path
        Output folder (``train_pool.csv``, ``test.csv``, ``folds.json``, ``split_summary.json``).
    published : Path or None
        Published ``split_manifest.csv`` that fixes the side of each primary patient.
    train_sources, test_sources : str
        Comma-separated sources allowed on each side (``lyria``, ``tufts``, ``dentex``).
    classes : str
        Comma-separated label space.
    cap : int
        Crops per class in the balanced training pool.
    class_caps : str
        Per-class overrides, e.g. ``Implant=93``.
    fill_class, fill_sources : str
        Class topped up on the training side from the given sources.
    n_folds, seed : int
        Folds and seed.

    Returns
    -------
    dict
        The split summary.

    Notes
    -----
    Groups never straddle a partition or a fold, the frozen holdout never enters training,
    and the primary partition reproduces the published one.
    """
    classes_l = classes.split(",")
    train_src = set(train_sources.split(","))
    test_src = set(test_sources.split(","))
    fill_src = set(s for s in fill_sources.split(",") if s)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    rows = [r for r in read_csv(manifest) if r["class"] in classes_l]
    for r in rows:
        r["frozen"] = int(r["frozen"])
    ly_tr, ly_te = lyria_partition(rows, cap, seed, published)

    def side(r: dict) -> str:
        """Return ``train``, ``test``, ``frozen`` or ``unused`` for one row."""
        s = r["source"]
        if r["frozen"]:
            return "frozen"
        if s == "lyria":
            if s in train_src and s not in test_src:
                return "train"
            if s in train_src and r["group"] in ly_tr:
                return "train"
            if s in test_src and r["group"] in ly_te:
                return "test"
            return "unused"
        if s in test_src:
            return "test"
        if s in train_src or (r["class"] == fill_class and s in fill_src):
            return "train"
        return "unused"

    for r in rows:
        r["side"] = side(r)
    if {r["group"] for r in rows if r["side"] == "train"} & {r["group"] for r in rows if r["side"] == "test"}:
        raise ValueError("patient leakage: a group is on both sides")

    pool_by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["side"] == "train":
            pool_by_class[r["class"]].append(r)
    caps = {}
    for kv in (x for x in class_caps.split(",") if x):
        k, v = kv.split("=")
        caps[k] = int(v)
    train_pool, short = [], {}
    for c in classes_l:
        c_cap = caps.get(c, cap)
        crops = pool_by_class[c][:]
        rng.shuffle(crops)
        if len(crops) < c_cap:
            short[c] = len(crops)
        train_pool.extend(crops[:c_cap])
    rng.shuffle(train_pool)
    if short:
        print(f"WARNING: below the cap of {cap}: {short}")
    test_set = [r for r in rows if r["side"] == "test"]

    cols = ["filepath", "class", "source", "patient_id", "group", "origin"]
    for name, data in (("train_pool.csv", train_pool), ("test.csv", test_set)):
        write_csv(out / name, [[r[c] if c != "patient_id" else r["group"] for c in cols] for r in data], cols)

    folds = group_folds(train_pool, "group", n_folds, seed, as_int=True)
    with open(out / "folds.json", "w") as fh:
        json.dump(folds, fh)
    groups = [r["group"] for r in train_pool]
    summary = {
        "train_sources": sorted(train_src),
        "test_sources": sorted(test_src),
        "classes": classes_l,
        "cap": cap,
        "class_caps": caps,
        "seed": seed,
        "fill": {"class": fill_class, "sources": sorted(fill_src)} if fill_class else None,
        "below_cap": short,
        "train_pool": dict(Counter(r["class"] for r in train_pool)),
        "train_pool_by_source": {
            f"{s}|{c}": n for (s, c), n in sorted(Counter((r["source"], r["class"]) for r in train_pool).items())
        },
        "test": dict(Counter(r["class"] for r in test_set)),
        "test_by_source": dict(Counter(r["source"] for r in test_set)),
        "train_groups": len(set(groups)),
        "test_groups": len({r["group"] for r in test_set}),
        "frozen_excluded": sum(1 for r in rows if r["side"] == "frozen"),
        "fold_val_sizes": {f: len(v["val"]) for f, v in folds.items()},
    }
    write_json(out / "split_summary.json", summary)
    return summary


def leakage_report(train: list[dict], test: list[dict], key: str = "patient_id") -> set:
    """Return the groups present on both sides (empty for a leakage-safe partition)."""
    return {r[key] for r in train} & {r[key] for r in test}


__all__ = [
    "make_split",
    "make_splits_clean",
    "lyria_partition",
    "group_folds",
    "greedy_patient_assignment",
    "leakage_report",
]
