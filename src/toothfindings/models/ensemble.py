"""Three-seed ensemble of test-mode models and the classification metrics shared by evaluations.

A configuration (enhancement, augmentation, backbone) is trained once per seed in test mode;
:class:`Ensemble` loads the resulting ``model.pt`` files and returns the softmax output of each
seed so that callers can average, threshold or re-weight them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ..constants import ARCH_INPUT, CLASSES4, IMAGENET_MEAN, IMAGENET_STD, TIMM_NAME


class Ensemble:
    """Test-mode models of one configuration, one per seed, returning per-seed softmax outputs.

    Parameters
    ----------
    exp : Path or None
        Experiment folder containing ``test/<enh>__<aug>__<arch>__seed<s>/model.pt``. Ignored
        when ``state_dicts`` is given.
    arch : str
        Backbone name, a key of :data:`~toothfindings.constants.ARCH_INPUT`.
    enh : str
        Enhancement variant (``"none"``, ``"msthgr"`` or ``"msthgr-clahe"``).
    aug : str
        Augmentation setting (``"noaug"`` or ``"aug"``).
    seeds : sequence of int
        Training seeds to load, in output order.
    classes : list of str, optional
        Label space of the models; the four-class space ``CLASSES4`` by default.
    device : str, optional
        Torch device; ``"cuda"`` when available, otherwise ``"cpu"``.
    state_dicts : list of dict, optional
        In-memory weights, one per seed (used by the tests); bypasses ``exp``.
    """

    def __init__(
        self,
        exp: Path | None,
        arch: str = "inception_v3",
        enh: str = "none",
        aug: str = "noaug",
        seeds=(0, 1, 2),
        classes: list[str] | None = None,
        device: str | None = None,
        state_dicts: list | None = None,
    ):
        import timm
        import torch
        import torchvision.transforms as T

        self.classes = list(classes or CLASSES4)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.arch = arch
        self.models = []
        for i, seed in enumerate(seeds):
            model = timm.create_model(TIMM_NAME.get(arch, arch), pretrained=False, num_classes=len(self.classes))
            if state_dicts is not None:
                state = state_dicts[i]
            else:
                weights_path = os.path.join(exp, "test", f"{enh}__{aug}__{arch}__seed{seed}", "model.pt")
                state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
            self.models.append(model.eval().to(self.device))
        size = ARCH_INPUT[arch]
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def predict(self, pil_list: list, batch: int = 64) -> np.ndarray:
        """Return the softmax probabilities of every seed for a list of PIL crops.

        Parameters
        ----------
        pil_list : list of PIL.Image.Image
            Crops to classify (converted by the evaluation transform of the backbone).
        batch : int
            Number of crops per forward pass.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(n_crops, n_seeds, n_classes)``.
        """
        import torch

        out = np.zeros((len(pil_list), len(self.models), len(self.classes)), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(pil_list), batch):
                x = torch.stack([self.tf(p) for p in pil_list[i : i + batch]]).to(self.device)
                for seed_idx, model in enumerate(self.models):
                    logits = model(x)
                    if isinstance(logits, (tuple, list)):  # InceptionV3 may also return auxiliary logits
                        logits = logits[0]
                    out[i : i + batch, seed_idx] = torch.softmax(logits.float(), 1).cpu().numpy()
        return out


def metrics(y_true, y_pred, classes: list[str] | None = None) -> dict:
    """Compute macro-F1, accuracy, per-class precision/recall/F1 and the confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        True and predicted class indices into ``classes``.
    classes : list of str, optional
        Label space; ``CLASSES4`` by default.

    Returns
    -------
    dict
        ``{"macro_f1": float, "accuracy": float, "per_class": {class: {"precision", "recall",
        "f1"}}, "confusion": list of lists}``. Undefined precision/recall count as 0
        (``zero_division=0``); confusion rows are true classes, columns predicted classes.
    """
    classes = list(classes or CLASSES4)
    labels = list(range(len(classes)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(f.mean()),
        "accuracy": float((np.array(y_true) == np.array(y_pred)).mean()),
        "per_class": {
            classes[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i in range(len(classes))
        },
        "confusion": cm.tolist(),
    }
