"""Mitigation variants of the training data.

These variants test two mitigations of false flags on sound teeth: wider crops that keep
anatomical context, and sound training negatives drawn by tooth position.

Four-class variants (:func:`build_variant`), reusing the canonical patient partition:

``ctx13``
    same pool and test set, every primary crop re-extracted with its box scaled x1.3 about the
    centre (context-preserving crop); the pre-cropped DENTEX implants are kept as they are.
``posneg``
    position-aware sound negatives: the 180 sound training crops are replaced by 60 third molars,
    60 second molars and 60 other sound teeth of the same training patients, patient-diverse;
    folds recomputed with ``StratifiedGroupKFold(5, shuffle, 0)``.
``posneg_ctx13``
    both.

Five-class mitigated variant (:func:`build_five_mitigated`): ``splits_clean5`` with the
position-aware negatives, x1.3 context crops for every crop (external ones re-extracted from
their radiograph with the recorded segmenter box), the evaluation manifests and the
tooth-position priors ``prior4.json``/``prior5.json`` (P(class | tooth type) estimated on the
training patients).
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..constants import CLASSES4, CLASSES5, POSNEG_TYPES, tooth_type
from ..utils import read_csv, write_csv, write_json
from .crops import gray3_save, round_robin
from .lyria_blocks import block_index, block_index5, parse_crop_name, scaled_box
from .splits import SPLIT_HEADER, group_folds, write_fold_csvs

SCALE = 1.3
FIVE_COLS = ["filepath", "class", "source", "patient_id", "group", "origin"]


# Four-class variants
def write_split4(out: Path, train_pool: list[dict], test: list[dict], seed: int = 0) -> None:
    """Write ``test.csv``, ``train_pool.csv``, ``folds.json`` and ``folds/`` of a four-class variant."""
    out = Path(out)
    for name, data in (("test.csv", test), ("train_pool.csv", train_pool)):
        write_csv(out / name, [[r["filepath"], r["class"], r["source"], r["patient_id"]] for r in data], SPLIT_HEADER)
    folds = group_folds(train_pool, "patient_id", 5, seed)
    write_fold_csvs(out, train_pool, folds)
    write_json(out / "folds.json", folds)


def select_posneg(cands: dict[str, list], rng: random.Random) -> list[tuple]:
    """Pick 60 third molars, 60 second molars and 60 other sound teeth, patient-diversely.

    Parameters
    ----------
    cands : dict
        Tooth type (``3rd molar``, ``2nd molar``, ``other``) -> candidate tuples whose first
        element is the patient id.
    rng : random.Random
        Shared generator, consumed type by type.

    Returns
    -------
    list of tuple
        The 180 chosen candidates.
    """
    chosen = []
    for t, need in (("3rd molar", 60), ("2nd molar", 60), ("other", 60)):
        by_patient: dict[str, list] = defaultdict(list)
        for item in cands[t]:
            by_patient[item[0]].append(item)
        selected = round_robin(by_patient, need, rng, with_pid=False)
        assert len(selected) == need, (t, len(selected))
        chosen.extend(selected)
    return chosen


def make_posneg_pool(paths: Paths, rng: random.Random) -> list[tuple]:
    """Draw position-aware sound negatives from the training patients of the canonical split.

    Returns
    -------
    list of tuple
        180 ``(pid, idx, box, W, H)`` sound blocks; see :func:`select_posneg`.
    """
    pool = read_csv(paths.split_dir("splits") / "train_pool.csv")
    train_pids = sorted({r["patient_id"] for r in pool if r["source"] == "lyria"})
    cands: dict[str, list] = {"3rd molar": [], "2nd molar": [], "other": []}
    for pid in train_pids:
        for (c, i), (box, code, W, H) in block_index(paths.lyria_json, paths.lyria_images, pid).items():
            if c != "Healthy":
                continue
            cands[tooth_type(code, POSNEG_TYPES, "other")].append((pid, i, box, W, H))
    print("sound candidates in training patients:", {k: len(v) for k, v in cands.items()})
    return select_posneg(cands, rng)


def build_variant(paths: Paths, name: str) -> None:
    """Build one four-class mitigation variant (``ctx13``, ``posneg`` or ``posneg_ctx13``).

    Writes ``results/splits/variants/<name>/splits/`` and the crops under
    ``data/crops/variants/<name>/enhanced/none/<Class>/``. Crops that the variant does not
    change are symlinked to the canonical grayscale crops instead of being re-extracted.
    """
    ctx, posneg = "ctx13" in name, "posneg" in name
    rng = random.Random(0)
    enh = paths.variant_crops(name) / "enhanced" / "none"
    src_enh_root = paths.crops_enhanced / "none"
    for c in CLASSES4:
        (enh / c).mkdir(parents=True, exist_ok=True)
    canon = paths.split_dir("splits")
    pool, test = read_csv(canon / "train_pool.csv"), read_csv(canon / "test.csv")
    if posneg:
        new_sound = make_posneg_pool(paths, rng)
        pool = [r for r in pool if r["class"] != "Healthy"]
        for pid, i, _box, _W, _H in new_sound:
            pool.append(
                {
                    "filepath": f"Healthy/lyria__{pid}__b{i}.png",
                    "class": "Healthy",
                    "source": "lyria",
                    "patient_id": pid,
                }
            )
        rng.shuffle(pool)
    rows = sorted(pool + test, key=lambda r: os.path.basename(r["filepath"]))
    cache: dict = {}
    n_new = n_link = 0
    for r in rows:
        fname = os.path.basename(r["filepath"])
        dst = enh / r["class"] / fname
        if dst.exists():
            continue
        src_enh = src_enh_root / r["class"] / fname
        if r["source"] != "lyria" or (not ctx and src_enh.exists()):
            os.symlink(src_enh.resolve(), dst)
            n_link += 1
            continue
        pid, idx = parse_crop_name(fname)
        if pid not in cache:  # rows are sorted by file name, so one patient's crops are contiguous
            cache = {pid: block_index(paths.lyria_json, paths.lyria_images, pid)}
        box, _code, W, H = cache[pid][(r["class"], idx)]
        if ctx:
            box = scaled_box(box, W, H, SCALE)
        gray3_save(Image.open(paths.lyria_image(pid)).crop(box), dst)
        n_new += 1
    write_split4(paths.split_dir(f"variants/{name}/splits"), pool, test)
    print(f"[{name}] pool={len(pool)} test={len(test)} new crops={n_new} linked={n_link}")


# Five-class mitigated variant
class _FiveBuilder:
    """Image access helpers of the five-class mitigated build (caches radiographs and boxes)."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.root = paths.variant_crops("five_mitigated")
        self.enh = self.root / "enhanced" / "none"
        self._ext: dict | None = None
        self._img: dict = {}

    def bi5(self, pid: str) -> dict:
        """Return :func:`~toothfindings.data.lyria_blocks.block_index5` of one primary patient."""
        return block_index5(self.paths.lyria_json, self.paths.lyria_images, pid)

    def ext_index(self) -> dict:
        """Map ``(source, crop_file)`` to ``(radiograph path, segmenter box)`` from the external predictions."""
        if self._ext is None:
            self._ext = {}
            for src in ("tufts", "dentex"):
                for r in read_csv(self.paths.external / src / "predictions.csv"):
                    self._ext[(src, r["crop_file"])] = (
                        self.paths.pan_root(src) / r["pan_file"],
                        tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2")),
                    )
        return self._ext

    def open_img(self, path: Path, lyria: bool) -> Image.Image:
        """Open a radiograph (cached): primary ones with PIL, external ones with cv2 + BGR->RGB.

        External radiographs are read with cv2 to match the external pipeline that produced
        the stored crops.
        """
        key = str(path)
        if key not in self._img:
            if len(self._img) > 8:
                self._img.clear()
            if lyria:
                im = Image.open(path)
                im.load()
            else:
                import cv2

                im = Image.fromarray(cv2.cvtColor(cv2.imread(key), cv2.COLOR_BGR2RGB))
            self._img[key] = im
        return self._img[key]

    def source_box(self, row: dict) -> tuple[Path, tuple, int, int, int | None, bool]:
        """Return ``(radiograph, tight box, W, H, fdi_code, is_primary)`` of a crop row.

        ``fdi_code`` is None for external crops.
        """
        if row["source"] == "lyria":
            pid, idx = parse_crop_name(row["filepath"])
            box, code, W, H = self.bi5(pid)[(row["class"], idx)]
            return self.paths.lyria_image(pid), box, W, H, code, True
        pan, box = self.ext_index()[(row["source"], row["filepath"])]
        W, H = self.open_img(pan, False).size
        return pan, box, W, H, None, False

    def extract(self, row: dict, scale: float, dst: Path) -> int:
        """Write the grayscale crop of ``row`` with its box scaled by ``scale``; return 1 if written, 0 if present."""
        if dst.exists():
            return 0
        pan, box, W, H, _, ly = self.source_box(row)
        dst.parent.mkdir(parents=True, exist_ok=True)
        gray3_save(self.open_img(pan, ly).crop(scaled_box(box, W, H, scale) if scale != 1.0 else box), dst)
        return 1

    def sanity_scale1(self, rows: list[dict], n: int = 300, seed: int = 0) -> dict:
        """Check that re-extracting at scale 1.0 reproduces the stored crops pixel for pixel (grayscale).

        Returns
        -------
        dict
            ``checked``, ``mismatch``, ``max_abs_diff`` and up to five mismatch ``examples``.
        """
        rng = random.Random(seed)
        sample = rng.sample(rows, min(n, len(rows)))
        bad, max_diff = [], 0
        for r in sample:
            stored = self.paths.crop_file(r["filepath"], r["source"])
            if not stored.exists():
                continue
            pan, box, W, H, _, ly = self.source_box(r)
            a = np.array(self.open_img(pan, ly).crop(box).convert("L")).astype(int)
            b = np.array(Image.open(stored).convert("L")).astype(int)
            if a.shape != b.shape:
                bad.append((r["filepath"], "shape", a.shape, b.shape))
                continue
            d = int(np.abs(a - b).max())
            max_diff = max(max_diff, d)
            if d:
                bad.append((r["filepath"], "maxdiff", d))
        return {"checked": len(sample), "mismatch": len(bad), "max_abs_diff": max_diff, "examples": bad[:5]}

    def posneg(self, pool: list[dict], rng: random.Random) -> tuple[list[tuple], dict]:
        """Draw position-aware negatives from the primary patients of ``pool``, as in :func:`make_posneg_pool`.

        Returns
        -------
        chosen : list of tuple
            180 ``(pid, idx, box, W, H, fdi_code)`` sound blocks.
        info : dict
            Candidate and selection counts for the build log.
        """
        train_pids = sorted({r["patient_id"].split(":", 1)[-1] for r in pool if r["source"] == "lyria"})
        cands: dict[str, list] = {"3rd molar": [], "2nd molar": [], "other": []}
        for pid in train_pids:
            for (c, i), (box, code, W, H) in self.bi5(pid).items():
                if c != "Healthy":
                    continue
                cands[tooth_type(code, POSNEG_TYPES, "other")].append((pid, i, box, W, H, code))
        chosen = select_posneg(cands, rng)
        info = {
            "train_patients": len(train_pids),
            "candidates": {k: len(v) for k, v in cands.items()},
            "chosen_patients": len({c[0] for c in chosen}),
            "chosen_by_type": dict(Counter(tooth_type(c[5]) for c in chosen)),
        }
        return chosen, info

    def make_pool(self, src_pool: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        """Replace the sound crops of ``src_pool`` with position-aware negatives; return ``(pool, info)``."""
        chosen, info = self.posneg(src_pool, rng)
        pool = [dict(r) for r in src_pool if r["class"] != "Healthy"]
        (self.root / "crops" / "Healthy").mkdir(parents=True, exist_ok=True)
        for pid, i, box, _W, _H, _code in chosen:
            fname = f"lyria__{pid}__b{i}.png"
            fpath = self.root / "crops" / "Healthy" / fname
            if not fpath.exists():
                Image.open(self.paths.lyria_image(pid)).crop(box).save(fpath)
            pool.append(
                {
                    "filepath": self.paths.relative_to_crops(fpath),
                    "class": "Healthy",
                    "source": "lyria",
                    "patient_id": f"lyria:{pid}",
                    "group": f"lyria:{pid}",
                    "origin": "primary_posneg",
                }
            )
        rng.shuffle(pool)
        return pool, info

    def estimate_prior(self, pids: list[str], classes: list[str]) -> dict:
        """Estimate P(class | tooth type) over all blocks of ``pids`` with Laplace (+1) smoothing.

        Returns
        -------
        dict
            ``classes``, ``n_patients``, ``prior`` (tooth type -> probabilities in ``classes``
            order) and the raw ``counts`` (tooth type -> class -> blocks).
        """
        counts_by_type: dict[str, Counter] = defaultdict(Counter)
        for pid in pids:
            for (c, _i), (_box, code, _W, _H) in self.bi5(pid).items():
                if c in classes:
                    counts_by_type[tooth_type(code)][c] += 1
        k = len(classes)
        prior = {
            t: [(cc[c] + 1) / (sum(cc[x] for x in classes) + k) for c in classes] for t, cc in counts_by_type.items()
        }
        counts = {t: {c: cc[c] for c in classes} for t, cc in counts_by_type.items()}
        return {"classes": classes, "n_patients": len(pids), "prior": prior, "counts": counts}


def write_split5(out: Path, pool: list[dict], test: list[dict], readme: str) -> dict:
    """Write a five-class split (``train_pool.csv``, ``test.csv``, ``folds.json``, summary, README).

    Returns
    -------
    dict
        The split summary, also written to ``split_summary.json``.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train_pool.csv", pool), ("test.csv", test)):
        write_csv(out / name, [[r.get(c, "") for c in FIVE_COLS] for r in data], FIVE_COLS)
    folds = group_folds(pool, "group", 5, 0, as_int=True)
    with open(out / "folds.json", "w") as fh:
        json.dump(folds, fh)
    if {r["group"] for r in pool} & {r["group"] for r in test}:
        raise ValueError("patient leakage between the training pool and the test set")
    groups = [r["group"] for r in pool]
    summary = {
        "train_pool": dict(Counter(r["class"] for r in pool)),
        "train_pool_by_source": {
            f"{s}|{c}": n for (s, c), n in sorted(Counter((r["source"], r["class"]) for r in pool).items())
        },
        "test": dict(Counter(r["class"] for r in test)),
        "test_by_source": dict(Counter(r["source"] for r in test)),
        "train_groups": len(set(groups)),
        "test_groups": len({r["group"] for r in test}),
        "fold_val_sizes": {f: len(v["val"]) for f, v in folds.items()},
        "fold_val_class_min": {f: min(Counter(pool[i]["class"] for i in v["val"]).values()) for f, v in folds.items()},
    }
    write_json(out / "split_summary.json", summary)
    (out / "README.txt").write_text(readme)
    return summary


README_IN = """splits5_mitigated: five-class split with position-aware sound negatives (built {date} by
toothfindings.data.variants.build_five_mitigated). Provenance:
  * test.csv is a verbatim copy of splits_clean5/test.csv, so every test patient keeps its side.
  * train_pool.csv: splits_clean5/train_pool.csv with its 180 sound crops replaced by 60 third molars,
    60 second molars and 60 other sound teeth (strict state H), drawn round-robin patient-diverse with
    random.Random(0) from the primary patients of that same training pool.
  * folds.json: StratifiedGroupKFold(5, shuffle=True, random_state=0), groups = patient or radiograph.
  * training reads every crop re-extracted with its box scaled x1.3 about the centre.
"""

README_LODO = """splits5_mitigated_lodo_lyria: leave-one-dataset-out split, InReDD-only training, with
position-aware sound negatives (built {date} by toothfindings.data.variants.build_five_mitigated).
  * test.csv is a verbatim copy of splits_lodo_lyria/test.csv (expert-reviewed, non-frozen crops).
  * train_pool.csv: splits_lodo_lyria/train_pool.csv with the 180 sound crops replaced by the
    position-aware negatives (random.Random(0)).
  * folds.json: StratifiedGroupKFold(5, shuffle=True, random_state=0) (only test mode is trained).
"""


def build_five_mitigated(paths: Paths) -> dict:
    """Build the five-class mitigated splits, x1.3 crops, evaluation manifests and priors.

    Writes ``splits5_mitigated`` and ``splits5_mitigated_lodo_lyria`` (LODO: leave-one-dataset-out,
    trained on the primary source only), and under ``variants/five_mitigated/`` the priors and
    ``eval_sets/{indomain_test,prevalence,external}.csv``. Each evaluation manifest gives the
    path of the tight crop (``path_s1.0``) and of the x1.3 context crop (``path_s1.3``).

    Returns
    -------
    dict
        Build log (also written to ``variants/five_mitigated/build_log.json``).
    """
    builder = _FiveBuilder(paths)
    vdir = paths.split_dir("variants/five_mitigated")
    date = time.strftime("%Y-%m-%d")
    log: dict = {}
    builder.enh.mkdir(parents=True, exist_ok=True)

    c5_pool = read_csv(paths.split_dir("splits_clean5") / "train_pool.csv")
    c5_test = read_csv(paths.split_dir("splits_clean5") / "test.csv")
    pool, info = builder.make_pool(c5_pool, random.Random(0))
    log["posneg_indomain"] = info
    log["split_indomain"] = write_split5(
        paths.split_dir("splits5_mitigated"), pool, c5_test, README_IN.format(date=date)
    )
    lo_pool = read_csv(paths.split_dir("splits_lodo_lyria") / "train_pool.csv")
    lo_test = read_csv(paths.split_dir("splits_lodo_lyria") / "test.csv")
    lpool, linfo = builder.make_pool(lo_pool, random.Random(0))
    log["posneg_lodo_lyria"] = linfo
    log["split_lodo_lyria"] = write_split5(
        paths.split_dir("splits5_mitigated_lodo_lyria"), lpool, lo_test, README_LODO.format(date=date)
    )

    # Priors from training-partition patients only.
    tr_pids = sorted({r["patient_id"].split(":", 1)[-1] for r in c5_pool if r["source"] == "lyria"})
    te_pids = {r["patient_id"].split(":", 1)[-1] for r in c5_test}
    if set(tr_pids) & te_pids:
        raise ValueError("patient leakage between the training pool and the test set")
    p5 = builder.estimate_prior(tr_pids, CLASSES5)
    p5["source"] = "lyria patients of splits_clean5/train_pool.csv"
    write_json(vdir / "prior5.json", p5)
    canon = read_csv(paths.split_dir("splits") / "train_pool.csv")
    c_pids = sorted({r["patient_id"] for r in canon if r["source"] == "lyria"})
    p4 = builder.estimate_prior(c_pids, CLASSES4)  # reproduces the four-class position prior
    p4["source"] = "lyria patients of splits/train_pool.csv (as the deployment-prevalence evaluation)"
    write_json(vdir / "prior4.json", p4)
    log["prior5_patients"], log["prior4_patients"] = len(tr_pids), len(c_pids)

    lyr = [r for r in c5_pool + c5_test if r["source"] == "lyria"]
    ext = [r for r in c5_pool + lo_test if r["source"] != "lyria"]
    log["sanity_scale1_lyria"] = builder.sanity_scale1(lyr)
    log["sanity_scale1_external"] = builder.sanity_scale1(ext)

    made = 0
    rows = pool + c5_test + lpool + lo_test
    rows.sort(key=lambda r: (r["source"], os.path.basename(r["filepath"])))
    for r in rows:
        made += builder.extract(r, SCALE, builder.enh / r["class"] / os.path.basename(r["filepath"]))
    log["enh_crops_written"] = made

    eval_dir = vdir / "eval_sets"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rel = paths.relative_to_crops
    # (a) in-domain test
    out_rows = []
    for r in c5_test:
        code = builder.source_box(r)[4]
        out_rows.append(
            [
                r["filepath"],
                rel(builder.enh / r["class"] / os.path.basename(r["filepath"])),
                r["class"],
                r["source"],
                r["patient_id"].split(":", 1)[-1],
                tooth_type(code),
            ]
        )
    write_csv(
        eval_dir / "indomain_test.csv",
        out_rows,
        ["path_s1.0", "path_s1.3", "class", "source", "patient_id", "tooth_type"],
    )
    # (b) deployment prevalence: the rows of the four-class evaluation in the same order, plus every
    #     restored block of the same primary test patients as a separate set
    test4 = read_csv(paths.split_dir("splits") / "test.csv")
    prev_rows, have = [], set()
    for r in test4:
        base = os.path.basename(r["filepath"])[:-4]
        prev_rows.append(
            {
                "path_s1.0": r["filepath"],
                "class": r["class"],
                "source": r["source"],
                "patient_id": r["patient_id"],
                "in_capped": 1,
                "set": "base",
            }
        )
        if r["class"] == "Healthy":
            have.add(base)
    pids = sorted({r["patient_id"] for r in test4 if r["source"] == "lyria"})
    full = paths.crops_prevalence / "Healthy"
    n_missing_prev = 0
    for pid in pids:
        bi = builder.bi5(pid)
        for c, idx in sorted([k for k in bi if k[0] == "Healthy"], key=lambda k: k[1]):
            base = f"lyria__{pid}__b{idx}"
            if base in have:
                continue
            fpath = full / f"{base}.png"
            if not fpath.exists():
                n_missing_prev += 1
                fpath = builder.root / "prevalence" / "s1.0" / "Healthy" / f"{base}.png"
                if not fpath.exists():
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(paths.lyria_image(pid)).crop(bi[(c, idx)][0]).save(fpath)
            prev_rows.append(
                {
                    "path_s1.0": rel(fpath),
                    "class": "Healthy",
                    "source": "lyria",
                    "patient_id": pid,
                    "in_capped": 0,
                    "set": "base",
                }
            )
    for pid in pids:
        bi = builder.bi5(pid)
        for c, idx in sorted([k for k in bi if k[0] == "Restored"], key=lambda k: k[1]):
            fpath = builder.root / "prevalence" / "s1.0" / "Restored" / f"lyria__{pid}__b{idx}.png"
            if not fpath.exists():
                fpath.parent.mkdir(parents=True, exist_ok=True)
                gray3_save(Image.open(paths.lyria_image(pid)).crop(bi[(c, idx)][0]), fpath)
            prev_rows.append(
                {
                    "path_s1.0": rel(fpath),
                    "class": "Restored",
                    "source": "lyria",
                    "patient_id": pid,
                    "in_capped": 0,
                    "set": "restored_extra",
                }
            )
    log["prevalence_missing_in_crops_prevalence"] = n_missing_prev
    made = 0
    out_rows = []
    for r in prev_rows:
        if r["source"] == "lyria":
            pid, idx = parse_crop_name(r["path_s1.0"])
            box, code, W, H = builder.bi5(pid)[(r["class"], idx)]
            s13 = builder.root / "prevalence" / "s1.3" / r["class"] / os.path.basename(r["path_s1.0"])
            if not s13.exists():
                s13.parent.mkdir(parents=True, exist_ok=True)
                gray3_save(builder.open_img(paths.lyria_image(pid), True).crop(scaled_box(box, W, H, SCALE)), s13)
                made += 1
            s13_rel, tt = rel(s13), tooth_type(code)
        else:  # pre-cropped DENTEX implants: unchanged, as the four-class evaluation at scale 1.3
            s13_rel, tt = r["path_s1.0"], "unknown"
        out_rows.append(
            [r["path_s1.0"], s13_rel, r["class"], r["source"], r["patient_id"], tt, r["in_capped"], r["set"]]
        )
    write_csv(
        eval_dir / "prevalence.csv",
        out_rows,
        ["path_s1.0", "path_s1.3", "class", "source", "patient_id", "tooth_type", "in_capped", "set"],
    )
    log["prevalence"] = {
        "rows": len(prev_rows),
        "by_set_class": {
            f"{s}|{c}": n for (s, c), n in sorted(Counter((r["set"], r["class"]) for r in prev_rows).items())
        },
        "s1.3_written": made,
        "lyria_patients": len(pids),
    }
    # (c) every expert-reviewed external crop, frozen flag kept
    from .manifests import manifest5_path

    rev = [r for r in read_csv(manifest5_path(paths)) if r["origin"] == "reviewed"]
    made = 0
    out_rows = []
    for r in rev:
        dst = builder.enh / r["class"] / os.path.basename(r["filepath"])
        made += builder.extract(r, SCALE, dst)
        out_rows.append([r["filepath"], rel(dst), r["class"], r["source"], r["group"], r["frozen"]])
    write_csv(eval_dir / "external.csv", out_rows, ["path_s1.0", "path_s1.3", "class", "source", "group", "frozen"])
    log["external"] = {
        "rows": len(rev),
        "by_source_class": {
            f"{s}|{c}": n for (s, c), n in sorted(Counter((r["source"], r["class"]) for r in rev).items())
        },
        "s1.3_written": made,
    }
    write_json(vdir / "build_log.json", log)
    return log


__all__ = ["build_variant", "build_five_mitigated", "make_posneg_pool", "select_posneg"]
