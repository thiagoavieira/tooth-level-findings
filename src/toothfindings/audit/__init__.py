"""Analysis of the expert review of external predictions (Tufts and DENTEX).

The review set is a stratified sample of the audited model's predictions. For each predicted
class: stratum A = score >= tau_c with three agreeing seeds (sampled), stratum B = score < tau_c
(50 sampled), stratum C = score >= tau_c without unanimity (not sampled, reported as
uncovered). Each reviewed crop stands for ``N_h / n_h`` predictions of its stratum; population
quantities are weighted ratio estimates with cluster bootstraps over radiographs.

Inputs: ``results/review_analysis/review_wide.csv`` (one row per reviewed crop, with both
readers' labels) and ``results/external/<ds>/{predictions,thresholds}_deit_posneg.*``.
"""
