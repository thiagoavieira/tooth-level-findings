"""Zero-shot pipeline on external panoramic radiographs.

Panoramic radiograph (PAN) -> mouth detection (RetinaNet, ONNX) -> mouth crop -> tooth instance
segmentation (YOLO11x-seg, ONNX) -> tooth boxes -> tight crop from the original radiograph ->
grayscale in three channels -> reference InceptionV3, softmax averaged over three seeds ->
class and probabilities.

Per dataset (``<results>/external/<ds>/``): ``predictions.csv`` (one row per tooth) and
``pans.csv`` (one row per radiograph); crops and mouth images are written under
``<data>/external/<ds>/``. Resumable: radiographs already in ``pans.csv`` are skipped.
Stored paths are relative (``pan_file`` to the dataset root, ``crop_file`` to the crop folder's
parent). The detector and segmenter belong to the InReDD platform and are not public.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..constants import CLASSES4, IMAGENET_MEAN, IMAGENET_STD
from ..data.crops import gray3

#: Input side of the reference classifier.
INPUT_SIZE = 299
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
#: Architecture of the reference classifier.
CLS_ARCH = "inception_v3"
#: Input size of the mouth detector.
MOUTH_W, MOUTH_H = 1920, 1080


def iter_images(root: Path):
    """Yield the image files under ``root`` (recursively, sorted), or ``root`` itself if it is a file."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


class MouthDetector:
    """RetinaNet mouth detector (ONNX), tried on CUDA first and on CPU if that fails.

    Parameters
    ----------
    weights : Path
        ONNX model.
    threshold : float
        Minimum detection score.
    """

    def __init__(self, weights: Path, threshold: float = 0.5):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 6
        so.inter_op_num_threads = 1
        try:
            self.sess = ort.InferenceSession(
                str(weights), sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception as e:
            print(f"[mouth] CUDA provider failed ({e}); using CPU", flush=True)
            self.sess = ort.InferenceSession(str(weights), sess_options=so, providers=["CPUExecutionProvider"])
        self.threshold = threshold

    def __call__(self, bgr: np.ndarray):
        """Detect the mouth in a BGR image.

        Returns
        -------
        tuple or None
            ``([x1, y1, x2, y2], score)`` of the highest-scoring valid detection, in image
            pixels, or ``None`` when no detection reaches the threshold.
        """
        import cv2

        h, w = bgr.shape[:2]
        x = cv2.resize(bgr, (MOUTH_W, MOUTH_H))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        boxes, scores, _ = self.sess.run(["boxes", "scores", "labels"], {"images": x})
        sx, sy = w / MOUTH_W, h / MOUTH_H
        best = None
        for b, s in zip(boxes, scores):
            s = float(s)
            if s < self.threshold:
                continue
            x1, y1 = max(0, int(b[0] * sx)), max(0, int(b[1] * sy))
            x2, y2 = min(w, int(b[2] * sx)), min(h, int(b[3] * sy))
            if x2 > x1 and y2 > y1 and (best is None or s > best[1]):
                best = ([x1, y1, x2, y2], s)
        return best


class ToothDetector:
    """YOLO11x-seg tooth segmenter (only the boxes are used).

    Parameters
    ----------
    weights : Path
        Model file loadable by ``ultralytics.YOLO``.
    threshold : float
        Minimum confidence.
    device : str
        Inference device.
    """

    def __init__(self, weights: Path, threshold: float = 0.5, device: str = "cuda:0"):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="segment")
        self.threshold = threshold
        self.device = device

    def __call__(self, bgr: np.ndarray):
        """Detect teeth in a BGR image; returns ``(boxes, confidences)`` as lists, boxes in xyxy pixels."""
        r = self.model.predict(source=bgr, conf=self.threshold, device=self.device, verbose=False, save=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [], []
        return r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()


class ToothClassifier:
    """Reference four-class classifier, one model per training seed.

    Parameters
    ----------
    run_dirs : list of Path
        Run folders, each with a ``model.pt`` state dict.
    device : str
        Preferred device; falls back to CPU when CUDA is unavailable.
    """

    def __init__(self, run_dirs: list[Path], device: str = "cuda"):
        import timm
        import torch
        import torchvision.transforms as T

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = []
        for run_dir in run_dirs:
            m = timm.create_model(CLS_ARCH, pretrained=False, num_classes=len(CLASSES4))
            state = torch.load(Path(run_dir) / "model.pt", map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            m.load_state_dict(state)
            m.eval().to(self.device)
            self.models.append(m)
        self.tf = T.Compose(
            [T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )

    def __call__(self, pil_crops: list) -> np.ndarray:
        """Return the per-seed softmax of grayscale-in-RGB PIL crops, shape ``(n, seeds, C)``."""
        import torch

        if not pil_crops:
            return np.zeros((0, len(self.models), len(CLASSES4)))
        x = torch.stack([self.tf(c) for c in pil_crops]).to(self.device)
        outs = []
        with torch.no_grad():
            for m in self.models:
                logits = m(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                outs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.stack(outs, axis=1)


def summarize_probs(pr: np.ndarray) -> dict:
    """Summarise the seed ensemble of one tooth.

    Parameters
    ----------
    pr : np.ndarray
        Per-seed probabilities, shape ``(seeds, C)``.

    Returns
    -------
    dict
        ``top`` (predicted class index, argmax of the mean), ``mean`` (mean probabilities),
        ``prob`` (mean probability of ``top``), ``agree`` (number of seeds whose own argmax is
        ``top``) and ``margin`` (gap between the two highest mean probabilities).
    """
    mean = pr.mean(axis=0)
    top = int(mean.argmax())
    sorted_desc = np.sort(mean)[::-1]
    return {
        "top": top,
        "mean": mean,
        "prob": float(mean[top]),
        "agree": int((pr.argmax(axis=1) == top).sum()),
        "margin": float(sorted_desc[0] - sorted_desc[1]),
    }


def run_pipeline(
    paths: Paths,
    dataset: str,
    images: Path,
    name_prefix: str = "",
    mouth_thr: float = 0.5,
    tooth_thr: float = 0.5,
    limit: int = 0,
) -> None:
    """Run the pipeline on every radiograph under ``images`` and append to the result CSVs.

    Parameters
    ----------
    paths : Paths
        Configured paths (radiographs, detector weights, classifier runs, outputs).
    dataset : {"tufts", "dentex"}
        External dataset.
    images : Path
        Folder (or single file) of radiographs, inside ``paths.pan_root(dataset)``.
    name_prefix : str
        Prefix of the radiograph ids, joined with ``__`` (e.g. the DENTEX subset name).
    mouth_thr, tooth_thr : float
        Score thresholds of the mouth detector and the tooth segmenter.
    limit : int
        Process only the first ``limit`` radiographs (0: all).

    Raises
    ------
    ValueError
        If ``images`` is not inside the configured dataset folder.
    """
    pan_root = paths.pan_root(dataset).resolve()
    if not Path(images).resolve().is_relative_to(pan_root):
        raise ValueError(f"{images} is not inside the configured {dataset} folder {pan_root}")
    import torch

    torch.set_num_threads(6)
    import cv2

    out = paths.external / dataset
    work = paths.external_work / dataset
    crops_dir, mouth_dir = work / "crops", work / "mouth"
    for d in (out, crops_dir, mouth_dir):
        d.mkdir(parents=True, exist_ok=True)
    pred_csv, pan_csv = out / "predictions.csv", out / "pans.csv"
    done = set()
    if pan_csv.exists():
        with open(pan_csv) as fh:
            done = {r["pan_id"] for r in csv.DictReader(fh)}
    mouth = MouthDetector(paths.mouth_onnx, mouth_thr)
    tooth = ToothDetector(paths.tooth_onnx, tooth_thr)
    seed_runs = [paths.experiments / "test" / f"none__noaug__{CLS_ARCH}__seed{s}" for s in range(3)]
    clf = ToothClassifier(seed_runs)
    pred_fields = (
        [
            "dataset",
            "pan_id",
            "pan_file",
            "tooth_idx",
            "crop_file",
            "x1",
            "y1",
            "x2",
            "y2",
            "det_score",
            "pred",
            "pred_prob",
            "seed_agreement",
            "margin",
        ]
        + [f"p_{c}" for c in CLASSES4]
        + [f"p_{c}_s{i}" for i in range(len(seed_runs)) for c in CLASSES4]
    )
    pan_fields = [
        "dataset",
        "pan_id",
        "pan_file",
        "width",
        "height",
        "mouth_x1",
        "mouth_y1",
        "mouth_x2",
        "mouth_y2",
        "mouth_score",
        "n_teeth",
        "sec",
    ]
    new_pred, new_pan = not pred_csv.exists(), not pan_csv.exists()
    with open(pred_csv, "a", newline="") as fp, open(pan_csv, "a", newline="") as fpan:
        wp = csv.DictWriter(fp, fieldnames=pred_fields)
        wpan = csv.DictWriter(fpan, fieldnames=pan_fields)
        if new_pred:
            wp.writeheader()
        if new_pan:
            wpan.writeheader()
        image_paths = list(iter_images(images))
        if limit:
            image_paths = image_paths[:limit]
        for img_path in image_paths:
            pan_id = (name_prefix + "__" if name_prefix else "") + img_path.stem
            if pan_id in done:
                continue
            t0 = time.time()
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"  ! unreadable: {img_path.name}", flush=True)
                continue
            H, W = bgr.shape[:2]
            # Without a mouth detection the whole radiograph is segmented (mouth_score 0).
            det = mouth(bgr)
            mbox, mscore = ([0, 0, W, H], 0.0) if det is None else det
            mx1, my1, mx2, my2 = mbox
            mouth_bgr = bgr[my1:my2, mx1:mx2]
            cv2.imwrite(str(mouth_dir / f"{pan_id}.jpg"), mouth_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            boxes, scores = tooth(mouth_bgr)
            pil_pan = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            crops, meta = [], []
            # Teeth numbered left to right; boxes are shifted from mouth to radiograph coordinates.
            for j, i in enumerate(sorted(range(len(boxes)), key=lambda i: boxes[i][0])):
                bx1, by1, bx2, by2 = boxes[i]
                x1, y1 = int(round(bx1)) + mx1, int(round(by1)) + my1
                x2, y2 = int(round(bx2)) + mx1, int(round(by2)) + my1
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crop = gray3(pil_pan.crop((x1, y1, x2, y2)))
                fname = f"{pan_id}__t{j:02d}.png"
                crop.save(crops_dir / fname)
                crops.append(crop)
                meta.append((j, fname, x1, y1, x2, y2, float(scores[i])))
            pan_rel = Path(img_path).resolve().relative_to(pan_root).as_posix()
            for (j, fname, x1, y1, x2, y2, det_score), pr in zip(meta, clf(crops)):
                s = summarize_probs(pr)
                row = {
                    "dataset": dataset,
                    "pan_id": pan_id,
                    "pan_file": pan_rel,
                    "tooth_idx": j,
                    "crop_file": f"crops/{fname}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "det_score": round(det_score, 4),
                    "pred": CLASSES4[s["top"]],
                    "pred_prob": round(s["prob"], 4),
                    "seed_agreement": s["agree"],
                    "margin": round(s["margin"], 4),
                }
                for c, v in zip(CLASSES4, s["mean"]):
                    row[f"p_{c}"] = round(float(v), 4)
                for si in range(pr.shape[0]):
                    for c, v in zip(CLASSES4, pr[si]):
                        row[f"p_{c}_s{si}"] = round(float(v), 4)
                wp.writerow(row)
            wpan.writerow(
                {
                    "dataset": dataset,
                    "pan_id": pan_id,
                    "pan_file": pan_rel,
                    "width": W,
                    "height": H,
                    "mouth_x1": mx1,
                    "mouth_y1": my1,
                    "mouth_x2": mx2,
                    "mouth_y2": my2,
                    "mouth_score": round(float(mscore), 4),
                    "n_teeth": len(meta),
                    "sec": round(time.time() - t0, 2),
                }
            )
            fp.flush()
            fpan.flush()
