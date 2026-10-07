"""Re-score the external teeth with another classifier and crop scale.

Reads ``external/<ds>/predictions.csv`` (boxes in radiograph coordinates), re-crops each box
from the radiograph enlarged by ``scale`` (grayscale in three channels), predicts with the
three seeds of ``<exp>/test/none__noaug__<arch>__seed{0,1,2}`` and writes
``predictions_<tag>.csv`` with the same columns. The audited model is DeiT-S trained with
position-aware negatives ("posneg"), at scale 1.0 (``tag = deit_posneg``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..constants import ARCH_INPUT, CLASSES4, IMAGENET_MEAN, IMAGENET_STD, TIMM_NAME
from ..data.crops import gray3
from ..data.lyria_blocks import scaled_box
from ..utils import read_csv
from .pipeline import summarize_probs


def rescore(
    paths: Paths,
    exp: Path,
    tag: str,
    datasets=("tufts", "dentex"),
    arch: str = "deit_small_patch16_224",
    scale: float = 1.0,
    batch: int = 32,
) -> None:
    """Re-score every tooth of each dataset and write ``external/<ds>/predictions_<tag>.csv``.

    Needs the radiographs and the trained weights. Rows are processed grouped by radiograph
    (sorted by ``pan_file``) and written in that order.

    Parameters
    ----------
    paths : Paths
        Configured paths.
    exp : Path
        Experiment folder holding ``test/none__noaug__<arch>__seed{0,1,2}/model.pt``.
    tag : str
        Suffix of the output file.
    datasets : sequence of str
        External datasets to re-score.
    arch : str
        Architecture name (key of ``ARCH_INPUT``; mapped to a timm name through ``TIMM_NAME``).
    scale : float
        Enlargement factor of each tooth box before cropping. The command line passes 1.0.
    batch : int
        Inference batch size.
    """
    import timm
    import torch
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for s in range(3):
        m = timm.create_model(TIMM_NAME.get(arch, arch), pretrained=False, num_classes=4)
        m.load_state_dict(
            torch.load(Path(exp) / "test" / f"none__noaug__{arch}__seed{s}" / "model.pt", map_location="cpu")
        )
        models.append(m.eval().to(device))
    size = ARCH_INPUT[arch]
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    for ds in datasets:
        rows = read_csv(paths.external / ds / "predictions.csv")
        rows.sort(key=lambda r: r["pan_file"])
        out = paths.external / ds / f"predictions_{tag}.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            current_pan, img, W, H = None, None, 0, 0
            crops: list = []
            pending_rows: list = []

            def flush():
                """Classify the buffered crops and write their rows."""
                if not crops:
                    return
                x = torch.stack([tf(c) for c in crops]).to(device)
                with torch.no_grad():
                    P = np.stack([torch.softmax(m(x), 1).cpu().numpy() for m in models], 1)
                for r, pr in zip(pending_rows, P):
                    s = summarize_probs(pr)
                    r["pred"] = CLASSES4[s["top"]]
                    r["pred_prob"] = f"{s['prob']:.4f}"
                    r["seed_agreement"] = str(s["agree"])
                    r["margin"] = f"{s['margin']:.4f}"
                    for c, v in zip(CLASSES4, s["mean"]):
                        r[f"p_{c}"] = f"{v:.4f}"
                    for si in range(3):
                        for c, v in zip(CLASSES4, pr[si]):
                            r[f"p_{c}_s{si}"] = f"{v:.4f}"
                    writer.writerow(r)
                crops.clear()
                pending_rows.clear()

            for r in rows:
                if r["pan_file"] != current_pan:
                    current_pan = r["pan_file"]
                    img = Image.open(paths.pan_root(ds) / current_pan).convert("L")
                    W, H = img.size
                box = scaled_box((int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])), W, H, scale)
                crops.append(gray3(img.crop(box)))
                pending_rows.append(dict(r))
                if len(crops) >= batch:
                    flush()
            flush()
