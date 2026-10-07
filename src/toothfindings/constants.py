"""Constants shared across the package: class names, backbones, run settings and tooth types.

Class names are the ones stored in the result files: ``Endodontics`` = endodontically treated
tooth, ``Healthy`` = sound tooth, ``Impacted`` = impacted tooth, ``Implant`` = dental implant,
``Restored`` = restored tooth (five-class study only).
"""

from __future__ import annotations

#: Canonical (alphabetical) order of the four classes; index order of every model output.
CLASSES4 = ["Endodontics", "Healthy", "Impacted", "Implant"]
#: Five-class label space with restored teeth.
CLASSES5 = ["Endodontics", "Healthy", "Impacted", "Implant", "Restored"]
#: Finding classes: a prediction of one of them on a sound tooth is a false flag.
FINDINGS = ["Endodontics", "Impacted", "Implant"]

#: Annotation state codes of the primary source (InReDD-PAN924) behind each pathological class.
PATHOLOGY = {"Implant": {"Im"}, "Impacted": {"M3i", "I"}, "Endodontics": {"TeM", "Te"}}
#: Precedence when one block carries several pathological states.
PRIORITY = ["Implant", "Impacted", "Endodontics"]

#: Per-channel normalisation statistics of ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#: Input resolution of each backbone.
ARCH_INPUT = {
    "resnetv2_50": 224,
    "vgg16": 224,
    "inception_v3": 299,
    "convnext_tiny": 224,
    "tf_efficientnetv2_s": 300,
    "deit_small_patch16_224": 224,
}
#: Batch size of each backbone in every published run.
BATCH = {
    "resnetv2_50": 64,
    "vgg16": 32,
    "inception_v3": 32,
    "convnext_tiny": 32,
    "tf_efficientnetv2_s": 32,
    "deit_small_patch16_224": 64,
}
#: Explicit ImageNet-1k pretraining tags of the recent backbones (fair against the in1k CNNs).
TIMM_NAME = {
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "tf_efficientnetv2_s": "tf_efficientnetv2_s.in1k",
    "deit_small_patch16_224": "deit_small_patch16_224.fb_in1k",
}

#: Crop enhancements: grayscale as is, MSTHGR (multi-scale top-hat by geodesic reconstruction)
#: and MSTHGR followed by CLAHE (contrast-limited adaptive histogram equalisation).
ENHANCEMENTS = ["none", "msthgr", "msthgr-clahe"]
#: Training without and with data augmentation.
AUGS = ["noaug", "aug"]
#: Classic backbones of the 18-configuration grid and the recent backbones compared with them.
CLASSIC_ARCHS = ["resnetv2_50", "vgg16", "inception_v3"]
NEW_ARCHS = ["convnext_tiny", "tf_efficientnetv2_s", "deit_small_patch16_224"]
#: Seeds of every published configuration and number of cross-validation folds.
SEEDS = [0, 1, 2]
FOLDS = 5

#: Coarse tooth type from the FDI unit digit (deployment prevalence, prior, five-class study).
TOOTH_TYPES = {
    1: "incisor",
    2: "incisor",
    3: "canine",
    4: "premolar",
    5: "premolar",
    6: "1st molar",
    7: "2nd molar",
    8: "3rd molar",
}
#: Fine tooth type used by the false-flags-by-position analysis.
TOOTH_TYPES_FINE = {
    1: "central incisor",
    2: "lateral incisor",
    3: "canine",
    4: "1st premolar",
    5: "2nd premolar",
    6: "1st molar",
    7: "2nd molar",
    8: "3rd molar",
}
#: Tooth types of the position-aware negatives (sound training crops drawn by tooth position):
#: second and third molars; every other tooth counts as "other".
POSNEG_TYPES = {7: "2nd molar", 8: "3rd molar"}


def config_id(enh: str, aug: str, arch: str) -> str:
    """Return the run-configuration identifier ``<enh>__<aug>__<arch>``."""
    return f"{enh}__{aug}__{arch}"


def tooth_type(code: int | None, table: dict[int, str] | None = None, default: str = "unknown") -> str:
    """Map an FDI tooth code to a tooth type via its unit digit.

    Parameters
    ----------
    code : int or None
        Two-digit FDI code (for example 38); ``None`` and 0 map to ``default``.
    table : dict, optional
        Unit digit -> type; :data:`TOOTH_TYPES` by default.
    default : str
        Type returned for unknown codes.

    Returns
    -------
    str
        Tooth type, e.g. ``"3rd molar"`` for 38.
    """
    table = TOOTH_TYPES if table is None else table
    return table.get(code % 10 if code else 0, default)
