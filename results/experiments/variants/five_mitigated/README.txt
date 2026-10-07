experiments/variants/five_mitigated: five-class DeiT-S with the four-class prevalence mitigations
(position-aware negatives + x1.3 context crops in training and test + tooth-position prior at inference).

runs/                 15 CV runs (5 folds x seeds 0,1,2) on splits5_mitigated
test/                 3 test-mode runs (seeds 0,1,2); test/none__noaug__deit_small_patch16_224__seed{0,1,2}/model.pt
                      are the deployable checkpoints (timm deit_small_patch16_224, num_classes=5, class order
                      Endodontics, Healthy, Impacted, Implant, Restored; input = x1.3 crop, grayscale x3,
                      224x224, ImageNet normalisation; then the prior of
                      variants/five_mitigated/prior5.json, weight P(c|t)*5, renormalise)
registry.csv, progress.json   run_matrix bookkeeping
eval/*.npz            per-seed softmax (no prior) of every model on every evaluation set (eval_five_mitigated.py)
results/results.json  every number of the report (analyze_five_mitigated.py)
results/results_reference_only_pretraining.json   reference-model numbers computed BEFORE training NEW
results/tables.txt    plain-text tables (make_five_mitigated_tables.py)
results/params.json   parameter counts, four- and five-class heads
LOG.txt               pre-registered decision rule, design decisions, thresholds, run notes
Sibling: experiments/variants/five_mitigated_lodo_lyria (InReDD-only LODO model, 3 test runs).
Commands: tools/queue_five_mitigated.sh ref|train|lodo|new, then analyze_five_mitigated.py and
make_five_mitigated_tables.py.
