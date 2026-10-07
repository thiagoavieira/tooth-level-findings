"""Run the experiment matrix with a resumable registry.

The matrix crosses 3 enhancements x 2 augmentation settings x 3 classic backbones = 18
configurations. Each configuration is trained 5 folds x 3 seeds in ``cv`` mode
(cross-validation on the training pool) and 3 seeds in ``test`` mode (full pool, evaluated on
the fixed test set). The three recent backbones add 3 configurations on grayscale crops
without augmentation (select them with ``archs``).

Runs whose ``metrics.json`` exists are skipped, so the matrix can be re-invoked until done.
Progress is kept in ``registry.csv`` (one row per attempted run, columns ``REG_COLS``) and
``progress.json`` of the experiment folder.
"""

from __future__ import annotations

import csv
import gc
import json
import time
import traceback
from pathlib import Path

from ..constants import AUGS, BATCH, CLASSIC_ARCHS, ENHANCEMENTS, FOLDS, SEEDS, config_id
from .train import DataConfig, RunConfig, run

#: Columns of ``registry.csv``.
REG_COLS = [
    "run_id",
    "mode",
    "enhancement",
    "aug",
    "arch",
    "fold",
    "seed",
    "status",
    "batch_size",
    "start_ts",
    "end_ts",
    "epochs_run",
    "best_val_macro_f1",
    "test_macro_f1",
    "peak_vram_MB",
    "run_dir",
    "error",
]


def build_tasks(exp: Path, enhs=None, augs=None, archs=None) -> list[dict]:
    """Enumerate the runs of the matrix in execution order.

    Per configuration, the cv runs come first (seed-major, then fold), followed by the test runs.

    Parameters
    ----------
    exp : Path
        Experiment folder (unused; ``run_dir`` values are relative to it).
    enhs, augs, archs : list of str, optional
        Sub-grid; ``ENHANCEMENTS``, ``AUGS`` and ``CLASSIC_ARCHS`` by default.

    Returns
    -------
    list of dict
        One dict per run with keys ``mode`` (``"cv"`` or ``"test"``), ``enh``, ``aug``,
        ``arch``, ``fold`` (-1 in test mode), ``seed``, ``run_id`` and ``run_dir``
        (``runs/<run_id>`` or ``test/<run_id>``).
    """
    tasks = []
    for enh in enhs or ENHANCEMENTS:
        for aug in augs or AUGS:
            for arch in archs or CLASSIC_ARCHS:
                cfg = config_id(enh, aug, arch)
                for seed in SEEDS:
                    for fold in range(FOLDS):
                        run_id = f"{cfg}__fold{fold}__seed{seed}"
                        tasks.append(
                            dict(
                                mode="cv",
                                enh=enh,
                                aug=aug,
                                arch=arch,
                                fold=fold,
                                seed=seed,
                                run_id=run_id,
                                run_dir=f"runs/{run_id}",
                            )
                        )
                for seed in SEEDS:
                    run_id = f"{cfg}__seed{seed}"
                    tasks.append(
                        dict(
                            mode="test",
                            enh=enh,
                            aug=aug,
                            arch=arch,
                            fold=-1,
                            seed=seed,
                            run_id=run_id,
                            run_dir=f"test/{run_id}",
                        )
                    )
    return tasks


def _epochs_run(run_dir: Path):
    """Return the number of epochs logged in ``training_log.csv`` (``""`` if absent)."""
    log = run_dir / "training_log.csv"
    if not log.exists():
        return ""
    with open(log) as fh:
        return max(0, sum(1 for _ in fh) - 1)


def _peak_vram(run_dir: Path):
    """Return the peak GPU memory (MB) recorded in ``status.json`` (``""`` if unavailable)."""
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            return json.loads(status_path.read_text()).get("gpu_mem_mb", "")
        except Exception:
            return ""
    return ""


def _write_progress(path: Path, total, done, failed, current, started) -> None:
    """Write ``progress.json``: counts, the current run, elapsed time and a linear ETA."""
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 and done else 0
    remaining = (total - done) / rate if rate else 0
    path.write_text(
        json.dumps(
            {
                "total": total,
                "done": done,
                "failed": failed,
                "current": current,
                "elapsed_s": round(elapsed),
                "eta_s": round(remaining),
                "pct": round(100 * done / total, 1),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )


def run_matrix(
    exp: Path,
    data: DataConfig,
    only: str = "all",
    epochs: int = 150,
    patience: int = 20,
    lr: float = 1e-4,
    workers: int = 6,
    max_runs: int = 0,
    enhancements=None,
    augs=None,
    archs=None,
) -> dict:
    """Run or resume the matrix, skipping runs that already have a ``metrics.json``.

    A failing run is recorded as ``failed`` in the registry (with the error message) and the
    matrix continues with the next run.

    Parameters
    ----------
    exp : Path
        Experiment folder; ``runs/``, ``test/``, ``registry.csv`` and ``progress.json`` are
        created inside it.
    data : DataConfig
        Splits, enhanced crops and label space.
    only : {"all", "cv", "test"}
        Restrict to one mode.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience in epochs without validation macro-F1 improvement.
    lr : float
        Adam learning rate.
    workers : int
        Data-loader worker processes.
    max_runs : int
        Stop after this many runs in this invocation (0: no cap).
    enhancements, augs, archs : list of str, optional
        Sub-grid; see :func:`build_tasks`.

    Returns
    -------
    dict
        ``{"done": int, "failed": int, "total": int}``, where ``done`` counts finished runs
        including those completed before this invocation.
    """
    import torch

    exp = Path(exp)
    (exp / "runs").mkdir(parents=True, exist_ok=True)
    (exp / "test").mkdir(parents=True, exist_ok=True)
    registry_path, progress_path = exp / "registry.csv", exp / "progress.json"
    tasks = build_tasks(exp, enhancements, augs, archs)
    if only != "all":
        tasks = [task for task in tasks if task["mode"] == only]
    total = len(tasks)
    registry = {}
    if registry_path.exists():
        with open(registry_path) as fh:
            registry = {r["run_id"]: r for r in csv.DictReader(fh)}
    started = time.time()
    done = sum(1 for task in tasks if (exp / task["run_dir"] / "metrics.json").exists())
    failed = ran = 0
    print(f"Matrix: {total} runs ({only}). Already done: {done}.")
    for task in tasks:
        run_dir = exp / task["run_dir"]
        if (run_dir / "metrics.json").exists():
            continue
        if max_runs and ran >= max_runs:
            print(f"Reached max_runs={max_runs}, stopping.")
            break
        _write_progress(progress_path, total, done, failed, task["run_id"], started)
        cfg = RunConfig(
            mode=task["mode"],
            enhancement=task["enh"],
            aug=task["aug"],
            arch=task["arch"],
            fold=task["fold"],
            seed=task["seed"],
            run_id=task["run_id"],
            run_dir=str(run_dir),
            batch=BATCH[task["arch"]],
            lr=lr,
            epochs=epochs,
            patience=patience,
            workers=workers,
        )
        row = dict.fromkeys(REG_COLS, "")
        row.update(
            run_id=task["run_id"],
            mode=task["mode"],
            enhancement=task["enh"],
            aug=task["aug"],
            arch=task["arch"],
            fold=task["fold"],
            seed=task["seed"],
            batch_size=BATCH[task["arch"]],
            run_dir=task["run_dir"],
            start_ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        t0 = time.time()
        try:
            result = run(cfg, data)
            row.update(
                status="done",
                best_val_macro_f1=round(result.get("best_val_macro_f1", 0), 4),
                test_macro_f1=(round(result["test_metrics"]["macro_f1"], 4) if "test_metrics" in result else ""),
                batch_size=result.get("batch_size", BATCH[task["arch"]]),
            )
            done += 1
        except Exception as e:
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
            failed += 1
            print(f"FAILED {task['run_id']}: {e}")
            traceback.print_exc()
        finally:
            row.update(
                end_ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                epochs_run=_epochs_run(run_dir),
                peak_vram_MB=_peak_vram(run_dir),
            )
            registry[task["run_id"]] = row
            with open(registry_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=REG_COLS)
                writer.writeheader()
                for run_id in sorted(registry):
                    writer.writerow(registry[run_id])
            ran += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            print(
                f"[{done}/{total}] {task['run_id']} {row['status']} val_f1={row['best_val_macro_f1']} "
                f"test_f1={row['test_macro_f1']} ({time.time() - t0:.0f}s)"
            )
    _write_progress(progress_path, total, done, failed, "", started)
    return {"done": done, "failed": failed, "total": total}


__all__ = ["run_matrix", "build_tasks", "BATCH"]
