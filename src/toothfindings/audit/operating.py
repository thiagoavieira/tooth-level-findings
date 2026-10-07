"""Operating points on external data with the expert audit as ground truth.

All quantities are weighted estimates over the reviewed crops (see :mod:`.core`); "detected"
means detected and cropped by the pipeline, so sensitivities are conditional on detection.

* :func:`operating_points`: per class and threshold, detection rate (analogue of the true
  acceptance rate, TAR), false-positive rate (false acceptance rate, FAR), false flags per
  radiograph and precision, with errors split into wrong class, outside the label space and
  unusable crop; the smallest thresholds reaching 90% and 95% external precision.
* :func:`bridge_metrics`: sensitivity, specificity, likelihood ratios and the equal-error point.
* :func:`fairness_view`: the impacted class charged three ways (classifier only, as deployed,
  end to end).
* :func:`impacted_recall`: where the teeth the experts called impacted sit, and sensitivity
  against the threshold with a 400-resample cluster bootstrap.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict

import numpy as np

from ..utils import read_csv, read_json, write_json
from .core import CLIN, DATASETS, N_PANS, UNUSABLE, AuditData, percentile_ci

#: Thresholds of the operating curves.
OP_GRID = [round(x, 3) for x in np.arange(0.30, 1.0, 0.01)] + [0.995, 0.999]
#: Target external precisions.
TARGETS = [0.90, 0.95]
FAIR_GRID = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
#: Detection recall of annotated impacted teeth: DENTEX check and the in-domain end-to-end evaluation.
DET_RECALL = {"dentex": 0.859}
IN_DOMAIN_DET_RECALL = 0.639
RECALL_GRID = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def operating_points(data: AuditData, write: bool = True) -> dict:
    """Compute the operating curves of each class (``operating_points.json``, ``curves_<ds>.csv``).

    The detection rate divides by the estimated number of true teeth of the class among the
    covered predictions (strata A and B); the false-positive rate divides by the other covered
    predictions.

    Returns
    -------
    dict
        Per dataset: number of radiographs, covered predictions and their fraction, estimated
        true teeth per class, and per class the study threshold, the smallest thresholds
        reaching each target precision on at least 20 kept predictions (``tau_ext_p90``,
        ``tau_ext_p95`` with the metrics there) and the point where miss rate and
        false-positive rate are closest (``equal_error_like``).
    """
    res = {}
    for ds in DATASETS:
        st = data.strata(ds)
        rows, disagree, _ = data.reviewed(ds)
        w = data.setup(ds)[2]
        # confusion[predicted][expert label] = estimated number of predictions
        confusion: dict = defaultdict(lambda: defaultdict(float))
        for r in rows:
            confusion[r["model_pred"]][r["label"]] += w[(r["model_pred"], r["stratum"])]
        covered = sum(sum(v.values()) for v in confusion.values())
        true_tot = {c: sum(confusion[m][c] for m in CLIN) for c in CLIN}
        curves = {}
        for c in CLIN:
            pts = []
            for t in OP_GRID:
                tp = fp_wrong = fp_other = fp_unusable = 0.0
                for r in rows:
                    if r["model_pred"] != c or float(r["pred_prob"]) < t:
                        continue
                    wi = w[(c, r["stratum"])]
                    if r["label"] == c:
                        tp += wi
                    elif r["label"] in UNUSABLE:
                        fp_unusable += wi
                    elif r["label"] == "Other":
                        fp_other += wi
                    else:
                        fp_wrong += wi
                fp = fp_wrong + fp_other + fp_unusable
                pts.append(
                    {
                        "tau": t,
                        "tp": tp,
                        "fp": fp,
                        "fp_wrong_class": fp_wrong,
                        "fp_outside_label_space": fp_other,
                        "fp_unusable_crop": fp_unusable,
                        "precision": (tp / (tp + fp)) if (tp + fp) else None,
                        "detection_rate": (tp / true_tot[c]) if true_tot[c] else None,
                        "false_positive_rate": (fp / (covered - true_tot[c])) if covered > true_tot[c] else None,
                        "flags_per_radiograph": fp / N_PANS[ds],
                        "kept": tp + fp,
                    }
                )
            curves[c] = pts
        chosen = {}
        for c in CLIN:
            entry: dict = {"tau_in_domain": st[c]["tau"]}
            for tgt in TARGETS:
                ok = [p for p in curves[c] if p["precision"] is not None and p["precision"] >= tgt and p["kept"] >= 20]
                entry[f"tau_ext_p{int(tgt * 100)}"] = min((p["tau"] for p in ok), default=None)
                if ok:
                    p = min(ok, key=lambda p: p["tau"])
                    entry[f"at_p{int(tgt * 100)}"] = {
                        k: p[k] for k in ("precision", "detection_rate", "flags_per_radiograph", "kept")
                    }
            best = min(
                (p for p in curves[c] if p["detection_rate"] is not None and p["false_positive_rate"] is not None),
                key=lambda p: abs((1 - p["detection_rate"]) - p["false_positive_rate"]),
                default=None,
            )
            if best:
                entry["equal_error_like"] = {
                    "tau": best["tau"],
                    "miss_rate": 1 - best["detection_rate"],
                    "false_positive_rate": best["false_positive_rate"],
                }
            chosen[c] = entry
        res[ds] = {
            "n_radiographs": N_PANS[ds],
            "covered_predictions": covered,
            "coverage_fraction": covered / sum(st[c]["n_pop"] for c in CLIN),
            "estimated_true_teeth": true_tot,
            "operating_points": chosen,
            "reader_disagreements_excluded": disagree,
        }
        if write:
            with open(data.paths.review / f"curves_{ds}.csv", "w", newline="") as fh:
                wcsv = csv.writer(fh)
                wcsv.writerow(
                    [
                        "dataset",
                        "class",
                        "tau",
                        "precision",
                        "detection_rate",
                        "false_positive_rate",
                        "flags_per_radiograph",
                        "tp",
                        "fp",
                        "fp_wrong_class",
                        "fp_outside_label_space",
                        "fp_unusable_crop",
                    ]
                )
                for c in CLIN:
                    for p in curves[c]:
                        wcsv.writerow(
                            [
                                ds,
                                c,
                                p["tau"],
                                p["precision"],
                                p["detection_rate"],
                                p["false_positive_rate"],
                                p["flags_per_radiograph"],
                                round(p["tp"], 1),
                                round(p["fp"], 1),
                                round(p["fp_wrong_class"], 1),
                                round(p["fp_outside_label_space"], 1),
                                round(p["fp_unusable_crop"], 1),
                            ]
                        )
    if write:
        write_json(data.paths.review / "operating_points.json", res)
    return res


def bridge_metrics(data: AuditData, write: bool = True) -> dict:
    """Express the operating curves as screening metrics (``bridge_metrics.json``).

    Reads ``operating_points.json`` and ``curves_<ds>.csv`` written by
    :func:`operating_points`. For each class, at the curve point nearest to the study threshold,
    to 0.90 and to 0.99: sensitivity (TAR), specificity, FAR, positive and negative likelihood
    ratios, precision (PPV) and false flags per radiograph; plus the equal-error point
    (``eer`` = mean of miss rate and FAR where they are closest).
    """
    op = read_json(data.paths.review / "operating_points.json")
    out: dict = {}
    for ds in DATASETS:
        cur: dict = {}
        for r in read_csv(data.paths.review / f"curves_{ds}.csv"):
            rec = {
                k: (float(v) if v not in ("", "None") else None) for k, v in r.items() if k not in ("dataset", "class")
            }
            cur.setdefault(r["class"], []).append(rec)
        out[ds] = {}
        for c in CLIN:
            pts = cur[c]
            rows = {}
            for name, t in (
                ("in_domain_tau", op[ds]["operating_points"][c]["tau_in_domain"]),
                ("tau_0.90", 0.90),
                ("tau_0.99", 0.99),
            ):
                p = min(pts, key=lambda p, t=t: abs(p["tau"] - t))
                sens, fpr = p["detection_rate"], p["false_positive_rate"]
                spec = 1 - fpr if fpr is not None else None
                rows[name] = {
                    "tau": p["tau"],
                    "sensitivity_TAR": sens,
                    "specificity": spec,
                    "false_positive_rate_FAR": fpr,
                    "lr_plus": (sens / fpr) if (sens is not None and fpr) else None,
                    "lr_minus": ((1 - sens) / spec) if (sens is not None and spec) else None,
                    "precision_PPV": p["precision"],
                    "flags_per_radiograph": p["flags_per_radiograph"],
                }
            eer = min(
                (p for p in pts if p["detection_rate"] is not None and p["false_positive_rate"] is not None),
                key=lambda p: abs((1 - p["detection_rate"]) - p["false_positive_rate"]),
            )
            rows["equal_error_point"] = {
                "tau": eer["tau"],
                "miss_rate": 1 - eer["detection_rate"],
                "false_positive_rate": eer["false_positive_rate"],
                "eer": (1 - eer["detection_rate"] + eer["false_positive_rate"]) / 2,
            }
            out[ds][c] = rows
    if write:
        write_json(data.paths.review / "bridge_metrics.json", out)
    return out


def fairness_view(data: AuditData, write: bool = True) -> dict:
    """Compute impacted sensitivity and precision under three accounting conventions (``fairness_view.json``).

    ``precision_classifier_only`` ignores flags on teeth outside the label space and on
    unusable crops, ``precision_as_deployed`` counts every flag, and the end-to-end
    sensitivities multiply the sensitivity given detection by a detector recall
    (``DET_RECALL`` on DENTEX, ``IN_DOMAIN_DET_RECALL`` in domain).
    """
    out = {}
    for ds in DATASETS:
        _st, rows, w, _tau = data.setup(ds)
        true_imp = sum(w[(r["model_pred"], r["stratum"])] for r in rows if r["label"] == "Impacted")
        table = {}
        for t in FAIR_GRID:
            tp = fp_all = fp_fair = 0.0
            for r in rows:
                if r["model_pred"] != "Impacted" or float(r["pred_prob"]) < t:
                    continue
                wi = w[("Impacted", r["stratum"])]
                if r["label"] == "Impacted":
                    tp += wi
                else:
                    fp_all += wi
                    if r["label"] not in UNUSABLE and r["label"] != "Other":
                        fp_fair += wi
            sens = tp / true_imp
            table[str(t)] = {
                "sensitivity_given_detection": sens,
                "precision_classifier_only": tp / (tp + fp_fair) if (tp + fp_fair) else None,
                "precision_as_deployed": tp / (tp + fp_all) if (tp + fp_all) else None,
                "end_to_end_sensitivity": sens * DET_RECALL[ds] if ds in DET_RECALL else None,
                "end_to_end_sensitivity_in_domain_detector": sens * IN_DOMAIN_DET_RECALL,
            }
        out[ds] = {"estimated_true_impacted_among_cropped_teeth": true_imp, "table": table}
    if write:
        write_json(data.paths.review / "fairness_view.json", out)
    return out


def impacted_recall(data: AuditData, n_boot: int = 400, write: bool = True) -> dict:
    """Locate the teeth the experts called impacted and trace sensitivity against the threshold.

    Writes ``impacted_recall.json``.

    Returns
    -------
    dict
        Per dataset: the study threshold, the estimated number of true impacted teeth, where
        they are (predicted impacted above or below tau, or predicted as another class), the
        below-tau sample, the size of the uncovered stratum C and, per threshold in
        ``RECALL_GRID``, sensitivity with a cluster-bootstrap CI, precision and false flags
        per radiograph.
    """
    res = {}
    for ds in DATASETS:
        st, rows, w, _tau = data.setup(ds)
        tau = st["Impacted"]["tau"]
        where: dict = defaultdict(float)
        raw: dict = defaultdict(int)
        for r in rows:
            if r["label"] != "Impacted":
                continue
            c = r["model_pred"]
            if c == "Impacted":
                k = (
                    "predicted impacted, score >= tau"
                    if float(r["pred_prob"]) >= tau
                    else "predicted impacted, score < tau"
                )
            else:
                k = f"predicted {c}"
            where[k] += w[(c, r["stratum"])]
            raw[k] += 1
        total = sum(where.values())
        nB = sum(1 for r in rows if r["model_pred"] == "Impacted" and r["stratum"] == "B")
        nB_imp = sum(
            1 for r in rows if r["model_pred"] == "Impacted" and r["stratum"] == "B" and r["label"] == "Impacted"
        )

        def stats(sample, t):
            # (sensitivity, precision, false flags per radiograph) of the impacted class at t.
            tp = fp = tot = 0.0
            for r in sample:
                c = r["model_pred"]
                wi = w[(c, r["stratum"])]
                if r["label"] == "Impacted":
                    tot += wi
                if c != "Impacted" or float(r["pred_prob"]) < t:
                    continue
                if r["label"] == "Impacted":
                    tp += wi
                else:
                    fp += wi
            return (tp / tot if tot else None, tp / (tp + fp) if (tp + fp) else None, fp / N_PANS[ds])

        # Cluster bootstrap over radiographs; one generator shared by all thresholds.
        rng = random.Random(0)
        groups: dict = defaultdict(list)
        for r in rows:
            groups[r["pan_id"]].append(r)
        keys = list(groups)
        curve = {}
        for t in RECALL_GRID:
            s0 = stats(rows, t)
            boots = []
            for _ in range(n_boot):
                sample: list = []
                for _ in range(len(keys)):
                    sample += groups[keys[rng.randrange(len(keys))]]
                v = stats(sample, t)
                if v[0] is not None:
                    boots.append(v[0])
            curve[str(t)] = {
                "sensitivity": s0[0],
                "sensitivity_ci95": percentile_ci(boots),
                "precision": s0[1],
                "flags_per_radiograph": s0[2],
            }
        res[ds] = {
            "tau_study": tau,
            "estimated_true_impacted": total,
            "where_they_are": {
                k: {"estimated": v, "share": v / total, "reviewed_crops": raw[k]}
                for k, v in sorted(where.items(), key=lambda kv: -kv[1])
            },
            "below_tau_sample": {
                "reviewed": nB,
                "labelled_impacted": nB_imp,
                "stratum_size": st["Impacted"]["B"],
                "weight_per_crop": st["Impacted"]["B"] / nB if nB else None,
            },
            "uncovered_stratum": st["Impacted"]["C_uncovered"],
            "curve": curve,
        }
    if write:
        write_json(data.paths.review / "impacted_recall.json", res)
    return res
