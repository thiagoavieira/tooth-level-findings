"""Grad-CAM attribution maps (class activation maps, CAM) of the tooth classifiers.

* :func:`gradcam_overlays`: qualitative success and failure overlays per class for one
  configuration, saved to ``experiments/gradcam/<config>/{success,failure}/``.
* :func:`gradcam_masks`: quantitative check against the tooth mask predicted by the tooth
  segmenter, for annotated teeth matched to a segmented tooth at box IoU >= 0.5. Measures the
  share of CAM mass inside the mask, the IoU of the top-20% CAM pixels with the mask, the
  pointing game (whether the CAM maximum falls inside the mask) and the mask-area baseline
  (share of the crop covered by the mask, i.e. the expected mass of a uniform map). The masks
  are model predictions, not manual annotations.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
from PIL import Image

from ..config import Paths
from ..constants import ARCH_INPUT, CLASSES4, IMAGENET_MEAN, IMAGENET_STD, TIMM_NAME
from ..data.lyria_blocks import iter_blocks, parse_crop_name
from ..utils import read_csv
from .robustness import iou


def target_conv_layer(model, num_classes: int = 3):
    """Return the convolution layer that Grad-CAM should target.

    This is the last ``Conv2d`` with more than ``num_classes`` output channels, which excludes a
    1x1 convolutional classifier head: targeting a timm 1x1 conv head yields an all-zero CAM.
    If no layer qualifies, the last ``Conv2d`` is returned.
    """
    import torch.nn as nn

    conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels > num_classes:
            conv = m
    if conv is None:
        conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    return conv


def load_image(path, size: int):
    """Load and resize a crop as RGB; return ``(array in [0, 1], ImageNet-normalised 1x3xHxW tensor)``.

    The resize is PIL's bicubic default, not the bilinear ``torchvision`` resize used in training
    and evaluation, so a prediction here can differ slightly from the released one.
    """
    import torch

    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    norm = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return arr, torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)


def cam_scores(g: np.ndarray, mask: np.ndarray, top: float = 0.2) -> tuple[float, float, int]:
    """Score a CAM against a tooth mask.

    Parameters
    ----------
    g : ndarray
        CAM; negative values are clipped to zero.
    mask : ndarray of bool
        Tooth mask, same shape as ``g``.
    top : float
        Fraction of highest CAM pixels used for the IoU.

    Returns
    -------
    tuple of (float, float, int)
        Share of CAM mass inside the mask, IoU of the top-``top`` CAM pixels with the mask,
        and the pointing hit (1 if the CAM maximum lies inside the mask).
    """
    g = np.clip(g, 0, None)
    total = g.sum() + 1e-9
    inside = float(g[mask].sum() / total)
    thr = np.quantile(g, 1 - top)
    top_mask = g >= thr
    iou_top = float((top_mask & mask).sum() / ((top_mask | mask).sum() + 1e-9))
    py, px = np.unravel_index(int(g.argmax()), g.shape)
    return inside, iou_top, int(mask[py, px])


def summarize_rows(rows: list[list]) -> dict:
    """Aggregate per-crop Grad-CAM rows into the ``summary_<arch>.json`` structure.

    Parameters
    ----------
    rows : list of list
        Rows in the column order of ``per_crop_<arch>.csv`` (``patient_id``, ``class``, ``idx``,
        ``seed``, ``pred``, ``correct``, ``box_iou``, ``mask_area_frac``, ``mass_inside``,
        ``iou_top20``, ``pointing``).

    Returns
    -------
    dict
        Mean scores over ``all``, ``correct`` and ``incorrect`` rows and per class (also split
        by correctness). The header fields ``arch``, ``n_rows`` and ``n_teeth_no_mask`` are
        added by the caller.
    """
    # columns: mask_area_frac, mass_inside, iou_top20, pointing, correct
    A = np.array([[r[7], r[8], r[9], r[10], r[5]] for r in rows], float)

    def summ(sel):
        a = A[sel]
        return (
            {
                "n": int(len(a)),
                "mask_area_frac": float(a[:, 0].mean()),
                "mass_inside": float(a[:, 1].mean()),
                "iou_top20": float(a[:, 2].mean()),
                "pointing": float(a[:, 3].mean()),
            }
            if len(a)
            else {"n": 0}
        )

    cls_arr = np.array([r[1] for r in rows])
    correct = A[:, 4] == 1
    return {
        "all": summ(np.ones(len(rows), bool)),
        "correct": summ(correct),
        "incorrect": summ(~correct),
        "per_class": {c: summ(cls_arr == c) for c in CLASSES4},
        "per_class_correct": {c: summ((cls_arr == c) & correct) for c in CLASSES4},
        "per_class_incorrect": {c: summ((cls_arr == c) & ~correct) for c in CLASSES4},
    }


def gradcam_overlays(paths: Paths, config: str, seed: int = 0, n: int = 4) -> None:
    """Save up to ``n`` correct and ``n`` misclassified Grad-CAM overlays per class.

    Needs the restricted images and the trained weights. Test crops are scanned in
    ``test.csv`` order until every quota is filled.

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    config : str
        Configuration ``<enhancement>__<augmentation>__<arch>`` (e.g. ``none__noaug__inception_v3``).
    seed : int
        Seed of the model to explain.
    n : int
        Overlays per (outcome, class).
    """
    import timm
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    enh, _aug, arch = config.split("__")
    size = ARCH_INPUT[arch]
    model_path = paths.experiments / "test" / f"{config}__seed{seed}" / "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(TIMM_NAME.get(arch, arch), pretrained=False, num_classes=len(CLASSES4))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    cam = GradCAM(model=model, target_layers=[target_conv_layer(model, len(CLASSES4))])
    out_base = paths.experiments / "gradcam" / config
    counts = {(k, c): 0 for k in ("success", "failure") for c in CLASSES4}
    for r in read_csv(paths.split_dir("splits") / "test.csv"):
        cls = r["class"]
        img_path = paths.crops_enhanced / enh / cls / os.path.basename(r["filepath"])
        if not img_path.exists():
            continue
        arr, tensor = load_image(img_path, size)
        with torch.no_grad():
            pred = int(model(tensor.to(device)).argmax(1).item())
        kind = "success" if pred == CLASSES4.index(cls) else "failure"
        if counts[(kind, cls)] >= n:
            continue
        overlay = show_cam_on_image(arr, cam(input_tensor=tensor.to(device))[0], use_rgb=True)
        (out_base / kind).mkdir(parents=True, exist_ok=True)
        name = f"{cls}_pred-{CLASSES4[pred]}_{os.path.splitext(os.path.basename(r['filepath']))[0]}.png"
        Image.fromarray(overlay).save(out_base / kind / name)
        counts[(kind, cls)] += 1
        if all(v >= n for v in counts.values()):
            break


def gradcam_masks(paths: Paths, arch: str = "inception_v3", limit: int = 0, top: float = 0.2) -> dict:
    """Score the Grad-CAM maps of the three seeds against segmenter tooth masks.

    Needs the restricted images, the trained weights and the ONNX mouth detector and tooth
    segmenter. Writes ``experiments/gradcam_masks/per_crop_<arch>.csv`` (one row per tooth and
    seed) and ``summary_<arch>.json``.

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    arch : str
        Backbone.
    limit : int
        Process only the first ``limit`` patients (0: all).
    top : float
        Fraction of highest CAM pixels used for the IoU.

    Returns
    -------
    dict
        The content of ``summary_<arch>.json``.
    """
    import cv2
    import timm
    import torch
    from pytorch_grad_cam import GradCAM
    from ultralytics import YOLO

    from ..external.pipeline import MouthDetector

    yolo = YOLO(str(paths.tooth_onnx), task="segment")
    mouth = MouthDetector(paths.mouth_onnx, 0.5)
    test = [r for r in read_csv(paths.split_dir("splits") / "test.csv") if r["source"] == "lyria"]
    want: dict[str, set] = defaultdict(set)
    for r in test:
        pid, idx = parse_crop_name(r["filepath"])
        want[pid].add((r["class"], idx))
    pids = sorted(want)[:limit] if limit else sorted(want)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = ARCH_INPUT[arch]
    models, cams = [], []
    for s in range(3):
        m = timm.create_model(TIMM_NAME.get(arch, arch), pretrained=False, num_classes=4)
        m.load_state_dict(
            torch.load(paths.experiments / "test" / f"none__noaug__{arch}__seed{s}" / "model.pt", map_location="cpu")
        )
        m.eval().to(device)
        models.append(m)
        cams.append(GradCAM(model=m, target_layers=[target_conv_layer(m, 4)]))
    mean = np.array(IMAGENET_MEAN, np.float32)
    std = np.array(IMAGENET_STD, np.float32)
    rows, n_nomask, t0 = [], 0, time.time()
    for k, pid in enumerate(pids):
        bgr = cv2.imread(str(paths.lyria_image(pid)))
        H, W = bgr.shape[:2]
        det = mouth(bgr)
        mx1, my1, mx2, my2 = det[0] if det else [0, 0, W, H]
        seg = yolo.predict(source=bgr[my1:my2, mx1:mx2], conf=0.5, device="cpu", verbose=False, save=False)[0]
        # segmented teeth, shifted from the mouth crop back to radiograph coordinates
        pboxes, polys = [], []
        if seg.boxes is not None and len(seg.boxes) and seg.masks is not None:
            for b, poly in zip(seg.boxes.xyxy.cpu().numpy(), seg.masks.xy):
                pboxes.append((b[0] + mx1, b[1] + my1, b[2] + mx1, b[3] + my1))
                polys.append(np.asarray(poly) + np.array([mx1, my1]))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for cls, idx, gbox in iter_blocks(paths.lyria_json, paths.lyria_images, pid, (W, H)):
            if (cls, idx) not in want[pid]:
                continue
            best_iou, best_i = 0.0, -1
            for i, pb in enumerate(pboxes):
                v = iou(gbox, pb)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_iou < 0.5:
                n_nomask += 1
                continue
            x1, y1, x2, y2 = gbox
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [polys[best_i].astype(np.int32)], 1)
            img = cv2.resize(gray[y1:y2, x1:x2], (size, size))
            crop_mask = cv2.resize(mask[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
            arr = np.stack([img] * 3, -1).astype(np.float32) / 255.0
            x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
            area = float(crop_mask.mean())
            for s, (m, cam) in enumerate(zip(models, cams)):
                with torch.no_grad():
                    pred = int(m(x).argmax(1))
                inside, iou_top, point = cam_scores(cam(input_tensor=x)[0], crop_mask, top)
                rows.append(
                    [
                        pid,
                        cls,
                        idx,
                        s,
                        CLASSES4[pred],
                        int(CLASSES4[pred] == cls),
                        round(best_iou, 3),
                        round(area, 4),
                        round(inside, 4),
                        round(iou_top, 4),
                        point,
                    ]
                )
        if (k + 1) % 50 == 0:
            print(f"[{k + 1}/{len(pids)}] {time.time() - t0:.0f}s rows={len(rows)} nomask={n_nomask}", flush=True)
    out = paths.experiments / "gradcam_masks"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"per_crop_{arch}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "patient_id",
                "class",
                "idx",
                "seed",
                "pred",
                "correct",
                "box_iou",
                "mask_area_frac",
                "mass_inside",
                "iou_top20",
                "pointing",
            ]
        )
        writer.writerows(rows)
    res = {"arch": arch, "n_rows": len(rows), "n_teeth_no_mask": n_nomask, **summarize_rows(rows)}
    with open(out / f"summary_{arch}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    return res
