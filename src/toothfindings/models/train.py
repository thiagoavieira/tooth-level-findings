"""Train one run of the experiment matrix (PyTorch + timm).

Two modes:

``cv`` (cross-validation)
    Train on the training indices of one fold and validate on its validation indices (used for
    the ablation and the statistical comparisons); reads ``train_pool.csv`` and ``folds.json``.
``test``
    Train on the full balanced pool, holding out a random ``max(3 * n_classes, n // 10)`` crops
    for early stopping, and evaluate on the fixed test set (``test.csv``); also saves the model,
    the confusion matrix and the predictions.

The enhancement variant selects the pre-computed image pool
(``<enhanced_root>/<variant>/<Class>/<file>``); grayscale images are replicated over three
channels and normalised with the ImageNet statistics. Written into ``run_dir``:
``run_config.yaml`` (JSON content despite the extension), ``split_used.csv``,
``training_log.csv``, ``status.json``, ``metrics.json`` and, in test mode,
``confusion_matrix.csv``, ``predictions.csv`` and ``model.pt``.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..constants import ARCH_INPUT, BATCH, CLASSES4, IMAGENET_MEAN, IMAGENET_STD, TIMM_NAME
from ..utils import read_csv


@dataclass
class RunConfig:
    """Arguments of one run (the fields written to ``run_config.yaml``).

    Attributes
    ----------
    mode : {"cv", "test"}
        Cross-validation fold run or final run evaluated on the test set.
    enhancement : {"none", "msthgr", "msthgr-clahe"}
        Image variant: grayscale crops, MSTHGR, or MSTHGR followed by CLAHE.
    aug : {"noaug", "aug"}
        Whether training uses data augmentation.
    arch : str
        Backbone, a key of :data:`~toothfindings.constants.ARCH_INPUT`.
    fold : int
        Fold index in cv mode, -1 in test mode.
    seed : int
        Seed of ``random``, NumPy and PyTorch, and of the test-mode validation hold-out.
    run_id : str
        Run identifier stored in ``metrics.json``.
    run_dir : str
        Output folder of the run.
    batch : int, optional
        Batch size; the published size of the backbone
        (:data:`~toothfindings.constants.BATCH`) by default. Halved on CUDA out-of-memory.
    lr : float
        Initial Adam learning rate.
    epochs : int
        Maximum number of epochs.
    patience : int
        Early-stopping patience in epochs without validation macro-F1 improvement.
    workers : int
        Data-loader worker processes.
    """

    mode: str
    enhancement: str
    aug: str
    arch: str
    fold: int = -1
    seed: int = 0
    run_id: str = "adhoc"
    run_dir: str = "."
    batch: int | None = None
    lr: float = 1e-4
    epochs: int = 150
    patience: int = 20
    workers: int = 6

    def __post_init__(self):
        if self.batch is None:
            self.batch = BATCH[self.arch]


@dataclass
class DataConfig:
    """Where a run reads its data from and which label space it uses.

    Attributes
    ----------
    splits : Path
        Folder with ``train_pool.csv``, ``folds.json`` and ``test.csv``.
    enhanced_root : Path
        Root of the enhancement variants (``<enhanced_root>/<variant>/<Class>/<file>``).
    classes : list of str
        Label space; sorted alphabetically to define the output index order.
    pretrained : bool
        Load ImageNet weights (True for every published run; False for offline tests).
    """

    splits: Path
    enhanced_root: Path
    classes: list[str] = field(default_factory=lambda: list(CLASSES4))
    pretrained: bool = True

    def __post_init__(self):
        self.classes = sorted(self.classes)


def enhanced_path(row: dict, variant: str, enhanced_root: Path) -> str:
    """Return the path of the crop of ``row`` in an enhancement variant, matched by file base name."""
    return os.path.join(enhanced_root, variant, row["class"], os.path.basename(row["filepath"]))


def build_transform(input_size: int, augment: bool):
    """Return the training transform (``augment=True``) or the evaluation transform (resize only)."""
    import torchvision.transforms as T

    if augment:
        return T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(12),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose([T.Resize((input_size, input_size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def make_dataset(rows: list[dict], variant: str, input_size: int, augment: bool, data: DataConfig):
    """Return a ``torch.utils.data.Dataset`` of ``(image tensor, class index)`` pairs.

    Parameters
    ----------
    rows : list of dict
        Split rows with at least ``filepath`` and ``class``.
    variant : str
        Enhancement variant to read the images from.
    input_size : int
        Square input size of the backbone.
    augment : bool
        Use the training augmentation.
    data : DataConfig
        Provides ``enhanced_root`` and the label space (index order).
    """
    from PIL import Image
    from torch.utils.data import Dataset

    class2idx = {c: i for i, c in enumerate(data.classes)}

    class CropDataset(Dataset):
        def __init__(self):
            self.items = [(enhanced_path(r, variant, data.enhanced_root), class2idx[r["class"]]) for r in rows]
            self.tf = build_transform(input_size, augment)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, y = self.items[i]
            return self.tf(Image.open(path).convert("RGB")), y

    return CropDataset()


def set_seed(seed: int) -> None:
    """Seed ``random``, NumPy and PyTorch (CPU and CUDA)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> dict:
    """Compute accuracy, macro-F1 and per-class precision/recall/F1 (undefined values count as 0).

    Returns
    -------
    dict
        ``{"accuracy": float, "macro_f1": float, "per_class": {class: {"precision", "recall",
        "f1"}}}``.
    """
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            classes[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i in range(len(classes))
        },
    }


def write_status(run_dir: str, **kw) -> None:
    """Write the keyword arguments, a timestamp and the peak GPU memory to ``run_dir/status.json``."""
    import torch

    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available():
        kw["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    with open(os.path.join(run_dir, "status.json"), "w") as fh:
        json.dump(kw, fh, indent=2)


def evaluate(model, loader, device, crit) -> tuple[list[int], list[int], float]:
    """Predict every batch of ``loader``; return true labels, predicted labels and the mean loss."""
    import torch

    with torch.no_grad():
        model.eval()
        ys, ps, loss_sum, n = [], [], 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss_sum += crit(out, y).item() * x.size(0)
            n += x.size(0)
            ps.extend(out.argmax(1).cpu().tolist())
            ys.extend(y.cpu().tolist())
    return ys, ps, loss_sum / max(n, 1)


def train_loop(model, train_loader, val_loader, device, run_dir, lr, max_epochs, patience, classes):
    """Train with Adam, mixed precision and early stopping on the validation macro-F1.

    The learning rate is halved by ``ReduceLROnPlateau`` (patience 7, minimum 1e-6) when the
    validation macro-F1 stops improving; training stops after ``patience`` consecutive epochs
    without improvement. One row per epoch is appended to ``run_dir/training_log.csv``.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train in place.
    train_loader, val_loader : torch.utils.data.DataLoader
        Training and validation data.
    device : torch.device
        Device the model lives on.
    run_dir : str
        Output folder of the run.
    lr : float
        Initial learning rate.
    max_epochs : int
        Maximum number of epochs.
    patience : int
        Early-stopping patience in epochs.
    classes : list of str
        Label space, in output index order.

    Returns
    -------
    best_metrics : dict
        Validation metrics of the best epoch (see :func:`compute_metrics`); the weights of that
        epoch are loaded back into ``model``.
    best_f1 : float
        Validation macro-F1 of the best epoch.
    """
    import torch
    import torch.nn as nn

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=7, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    log_path = os.path.join(run_dir, "training_log.csv")
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "epoch_time_s"]
        )
    best_f1, best_state, best_metrics, epochs_without_improvement = -1.0, None, None, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        ys, ps, loss_sum, n = [], [], 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            ps.extend(out.argmax(1).detach().cpu().tolist())
            ys.extend(y.cpu().tolist())
        train_loss = loss_sum / max(n, 1)
        train_acc = accuracy_score(ys, ps)
        val_true, val_pred, val_loss = evaluate(model, val_loader, device, crit)
        val_metrics = compute_metrics(val_true, val_pred, classes)
        val_f1 = val_metrics["macro_f1"]
        sched.step(val_f1)
        cur_lr = opt.param_groups[0]["lr"]
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{train_acc:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_metrics['accuracy']:.4f}",
                    f"{val_f1:.4f}",
                    f"{cur_lr:.2e}",
                    f"{time.time() - t0:.2f}",
                ]
            )
        write_status(
            run_dir,
            state="running",
            epoch=epoch,
            total_epochs=max_epochs,
            best_val_macro_f1=round(best_f1, 4),
            last_val_macro_f1=round(val_f1, 4),
        )
        if val_f1 > best_f1:
            best_f1, best_metrics, epochs_without_improvement = val_f1, val_metrics, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, best_f1


def build_model(arch: str, n_classes: int, pretrained: bool = True, device=None):
    """Create a timm backbone with ``n_classes`` outputs, moved to ``device`` when given.

    The recent backbones use their ImageNet-1k weight tags (see ``TIMM_NAME``).
    """
    import timm

    model = timm.create_model(TIMM_NAME.get(arch, arch), pretrained=pretrained, num_classes=n_classes)
    return model.to(device) if device is not None else model


def run(args: RunConfig, data: DataConfig) -> dict:
    """Train and evaluate one run and write its outputs (see the module docstring).

    On CUDA out-of-memory the run restarts with half the batch size; the error is re-raised
    once the batch size is 4 or less. In test mode the early-stopping hold-out is a random 10%
    of the training pool, drawn by crop rather than by patient. GPU training is not bit-for-bit
    deterministic: deterministic cuDNN algorithms are not enforced and training uses mixed precision.

    Parameters
    ----------
    args : RunConfig
        Run arguments.
    data : DataConfig
        Data locations and label space.

    Returns
    -------
    dict
        Content of ``metrics.json``.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    classes = data.classes
    os.makedirs(args.run_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = ARCH_INPUT[args.arch]
    with open(os.path.join(args.run_dir, "run_config.yaml"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    write_status(args.run_dir, state="starting", device=str(device))

    pool = read_csv(Path(data.splits) / "train_pool.csv")
    if args.mode == "cv":
        with open(Path(data.splits) / "folds.json") as fh:
            fold = json.load(fh)[str(args.fold)]
        train_rows = [pool[i] for i in fold["train"]]
        val_rows = [pool[i] for i in fold["val"]]
    else:
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        n_val = max(len(classes) * 3, len(pool) // 10)
        val_rows, train_rows = pool[:n_val], pool[n_val:]

    with open(os.path.join(args.run_dir, "split_used.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filepath", "class", "role"])
        for r in train_rows:
            w.writerow([r["filepath"], r["class"], "train"])
        for r in val_rows:
            w.writerow([r["filepath"], r["class"], "val"])

    train_ds = make_dataset(train_rows, args.enhancement, input_size, args.aug == "aug", data)
    val_ds = make_dataset(val_rows, args.enhancement, input_size, False, data)
    batch = args.batch
    while True:
        try:
            train_loader = DataLoader(
                train_ds, batch_size=batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False
            )
            val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=args.workers, pin_memory=True)
            model = build_model(args.arch, len(classes), data.pretrained, device)
            best_metrics, best_f1 = train_loop(
                model, train_loader, val_loader, device, args.run_dir, args.lr, args.epochs, args.patience, classes
            )
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 4:
                raise
            batch //= 2
            print(f"OOM -> retry with batch={batch}")

    result = {
        "run_id": args.run_id,
        "mode": args.mode,
        "enhancement": args.enhancement,
        "aug": args.aug,
        "arch": args.arch,
        "fold": args.fold,
        "seed": args.seed,
        "batch_size": batch,
        "best_val_macro_f1": best_f1,
        "val_metrics": best_metrics,
    }
    if args.mode == "test":
        test_rows = read_csv(Path(data.splits) / "test.csv")
        test_ds = make_dataset(test_rows, args.enhancement, input_size, False, data)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=args.workers)
        test_true, test_pred, _ = evaluate(model, test_loader, device, nn.CrossEntropyLoss())
        result["test_metrics"] = compute_metrics(test_true, test_pred, classes)
        cm = confusion_matrix(test_true, test_pred, labels=list(range(len(classes))))
        with open(os.path.join(args.run_dir, "confusion_matrix.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["", *classes])
            for i, c in enumerate(classes):
                w.writerow([c, *cm[i].tolist()])
        with open(os.path.join(args.run_dir, "predictions.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filepath", "source", "y_true", "y_pred"])
            for r, yt, yp in zip(test_rows, test_true, test_pred):
                w.writerow([r["filepath"], r["source"], classes[yt], classes[yp]])
        torch.save(model.state_dict(), os.path.join(args.run_dir, "model.pt"))

    with open(os.path.join(args.run_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    write_status(
        args.run_dir,
        state="done",
        best_val_macro_f1=round(best_f1, 4),
        test_macro_f1=result.get("test_metrics", {}).get("macro_f1"),
    )
    print(
        f"[{args.run_id}] done. val_macro_f1={best_f1:.4f}"
        + (f" test_macro_f1={result['test_metrics']['macro_f1']:.4f}" if args.mode == "test" else "")
    )
    return result
