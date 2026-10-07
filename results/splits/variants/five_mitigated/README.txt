variants/five_mitigated: crops and evaluation manifests of the five-class mitigated DeiT-S study
(built 2026-10-02 by tools/make_five_mitigated.py; nothing pre-existing was modified).

enhanced/none/<Class>/<basename>.png
    Every training and test crop of splits5_mitigated and splits5_mitigated_lodo_lyria,
    plus every expert-reviewed external crop, re-extracted from its source radiograph with the box
    scaled x1.3 about its centre (clipped to the image) and saved as grayscale replicated to 3 channels.
    Lyria boxes: annotation blocks (lyria_blocks.py / generate_extra_crops.py indexing). External boxes:
    the segmenter box recorded in datasets/iei_external/<tufts|dentex>/predictions.csv (radiograph read
    with cv2 as external_pipeline/run_pipeline.py). train.py reads this tree via IEI_ENHANCED.
crops/Healthy/
    Tight crops of the position-aware sound negatives (provenance only; training uses the x1.3 versions).
prevalence/s1.0, prevalence/s1.3
    Deployment-prevalence crops of the 586 lyria test patients: x1.3 versions of the 8,672 crops of
    eval_prevalence.py, and the 4,919 restored teeth of the same patients (tight and x1.3).
eval_sets/indomain_test.csv   splits_clean5 test set (3,017 crops) with tooth type and both crop scales
eval_sets/prevalence.csv      8,672 base rows (same order as eval_prevalence.build_rows) + 4,919 restored rows
eval_sets/external.csv        2,283 expert-reviewed Tufts/DENTEX crops (frozen flag kept)
prior5.json   P(c|t), five classes, lyria patients of splits_clean5/train_pool.csv (195), Laplace +1
prior4.json   P(c|t), four classes, lyria patients of splits/train_pool.csv (193), = eval_prevalence prior
build_log.json  counts and sanity checks (tight re-extraction reproduces stored crops pixel for pixel)
