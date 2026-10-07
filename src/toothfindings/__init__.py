"""Tooth-level findings on panoramic radiographs: research code and released results.

Sub-packages
------------
data        crops, manifests, partitions and mitigation variants (need the restricted images)
enhance     MSTHGR and CLAHE enhancement
models      training, run matrix, three-seed ensemble, inference cost
evaluation  deployment prevalence, operating points, robustness, Grad-CAM, restored class
stats       summaries, Friedman/Nemenyi, paired tests with Holm, Cohen's d_z, bootstrap
external    Tufts/DENTEX pipeline, rescoring, study thresholds, blind review, DENTEX checks
audit       expert-review analysis
reporting   tables and figures recomputed from ``results/``
"""

__version__ = "1.0.0"
