import os
import random
import subprocess
import sys
from collections import Counter

import pytest

from toothfindings.cli import CLEAN_SPLITS
from toothfindings.data.splits import (
    CLASSES,
    greedy_patient_assignment,
    group_folds,
    leakage_report,
    lyria_partition,
    make_split,
    make_splits_clean,
)
from toothfindings.utils import read_csv, read_json, write_csv

RELEASED = [
    "splits",
    "variants/posneg/splits",
    "variants/ctx13/splits",
    "variants/posneg_ctx13/splits",
    "splits_clean4",
    "splits_clean5",
    "splits_lodo_lyria",
    "splits_lodo_dentex",
    "splits_lodo_tufts",
    "splits5_mitigated",
    "splits5_mitigated_lodo_lyria",
]


def assert_leakage_safe(pool, test, folds, key):
    assert not leakage_report(pool, test, key)
    val = []
    for f in folds.values():
        assert sorted(f["train"] + f["val"]) == list(range(len(pool)))
        assert not leakage_report([pool[i] for i in f["train"]], [pool[i] for i in f["val"]], key)
        val += f["val"]
    assert sorted(val) == list(range(len(pool)))


@pytest.mark.parametrize("name", RELEASED)
def test_released_partitions_are_leakage_safe(paths, name):
    d = paths.splits / name
    pool, test = read_csv(d / "train_pool.csv"), read_csv(d / "test.csv")
    key = "group" if "group" in pool[0] else "patient_id"
    assert_leakage_safe(pool, test, read_json(d / "folds.json"), key)


def synthetic_manifest(path, n_patients=80, seed=1):
    rng = random.Random(seed)
    rows = []
    for p in range(1, n_patients + 1):
        pid = f"P{p:04d}"
        for c in CLASSES:
            for b in range(rng.choice([0, 0, 1, 2, 3])):
                rows.append([f"{c}/lyria__{pid}__b{b}.png", c, "lyria", pid, 100, 200])
    write_csv(path, rows, ["filepath", "class", "source", "patient_id", "width", "height"])
    return rows


def test_greedy_assignment_stops_using_a_patient_once_its_classes_are_full():
    by_patient = {
        "A": [{"class": "Implant"}] * 2,
        "B": [{"class": "Implant"}, {"class": "Healthy"}],
        "C": [{"class": "Implant"}],
    }
    tr, te, counts = greedy_patient_assignment(by_patient, ["A", "B", "C"], n_train=2)
    assert (tr, te) == ({"A", "B"}, {"C"})
    assert counts == Counter({"Implant": 3, "Healthy": 1})


def test_group_folds_are_stratified_and_group_disjoint():
    pool = [{"class": c, "pid": f"p{i // 2}"} for i, c in enumerate(["a", "b"] * 30)]
    folds = group_folds(pool, "pid", n_folds=5, seed=0)
    assert_leakage_safe(pool, [], folds, "pid")
    assert all(isinstance(i, int) for i in group_folds(pool, "pid", as_int=True)["0"]["val"])


def test_make_split_on_a_synthetic_manifest(tmp_path):
    rows = synthetic_manifest(tmp_path / "manifest.csv")
    summary = make_split(tmp_path / "manifest.csv", tmp_path / "out", n_train=6, n_folds=3)
    out = tmp_path / "out"
    pool, test = read_csv(out / "train_pool.csv"), read_csv(out / "test.csv")
    assert Counter(r["class"] for r in pool) == {c: 6 for c in CLASSES}
    assert len(test) == summary["total_test_crops"]
    assert_leakage_safe(pool, test, read_json(out / "folds.json"), "patient_id")
    # every crop of a test patient is in the test set; the surplus of training patients is dropped
    test_pids = {r["patient_id"] for r in test}
    assert len(test) == sum(1 for r in rows if r[3] in test_pids)
    assert len(read_csv(out / "split_manifest.csv")) == len(pool) + len(test)
    for f in range(3):
        assert len(read_csv(out / "folds" / f"fold{f}_val.csv")) == summary["fold_val_sizes"][str(f)]


_RERUN = """
import sys
from pathlib import Path
from toothfindings.data.splits import make_split
make_split(Path(sys.argv[1]), Path(sys.argv[2]), n_train=6, n_folds=3)
"""


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="make_split iterates a set of patient ids")
def test_make_split_is_reproducible_across_interpreter_runs(tmp_path):
    synthetic_manifest(tmp_path / "manifest.csv")
    pools = []
    for hash_seed in ("0", "1"):
        out = tmp_path / f"run{hash_seed}"
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        subprocess.run([sys.executable, "-c", _RERUN, tmp_path / "manifest.csv", out], check=True, env=env)
        pools.append((out / "train_pool.csv").read_text())
    assert pools[0] == pools[1]


def test_lyria_partition_keeps_the_published_side(tmp_path):
    published = tmp_path / "split_manifest.csv"
    write_csv(
        published,
        [["Implant/lyria__P0001__b0.png", "Implant", "lyria", "P0001", "train_pool"]]
        + [["Healthy/lyria__P0002__b0.png", "Healthy", "lyria", "P0002", "test"]],
        ["filepath", "class", "source", "patient_id", "partition"],
    )
    rows = [{"source": "lyria", "group": f"lyria:P000{i}"} for i in range(1, 9)]
    tr, te = lyria_partition(rows, 180, 0, published)
    assert "lyria:P0001" in tr and "lyria:P0002" in te
    assert tr | te == {r["group"] for r in rows} and not tr & te
    assert len(tr) == 1 + 2  # the six unseen patients: one in three to training


def _clean_split(paths, tmp_path, name):
    out = tmp_path / name
    make_splits_clean(
        paths.splits / "crops_source" / "manifest_clean.csv",
        out,
        paths.splits / "splits" / "split_manifest.csv",
        **CLEAN_SPLITS[name],
    )
    return out


def _filepaths(path):
    return [r["filepath"] for r in read_csv(path)]


@pytest.mark.parametrize("name", ["splits_lodo_lyria", "splits_lodo_dentex", "splits_lodo_tufts"])
def test_lodo_partitions_are_rebuilt_from_the_released_manifest(paths, tmp_path, name):
    out = _clean_split(paths, tmp_path, name)
    for f in ("train_pool.csv", "test.csv"):
        assert _filepaths(out / f) == _filepaths(paths.splits / name / f)


@pytest.mark.parametrize("name", ["splits_clean4", "splits_clean5"])
def test_clean_partitions_keep_every_published_patient_on_its_side(paths, tmp_path, name):
    out = _clean_split(paths, tmp_path, name)
    published = {r["patient_id"]: r["partition"] for r in read_csv(paths.splits / "splits" / "split_manifest.csv")}
    for f, side in (("train_pool.csv", "train_pool"), ("test.csv", "test")):
        for rows in (read_csv(out / f), read_csv(paths.splits / name / f)):
            pids = {r["group"].split(":")[1] for r in rows if r["source"] == "lyria"}
            assert all(published.get(p, side) == side for p in pids)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="splits_clean4/5 predate the released manifest_clean.csv")
@pytest.mark.parametrize("name", ["splits_clean4", "splits_clean5"])
def test_clean_partitions_are_rebuilt_from_the_released_manifest(paths, tmp_path, name):
    out = _clean_split(paths, tmp_path, name)
    assert _filepaths(out / "test.csv") == _filepaths(paths.splits / name / "test.csv")
