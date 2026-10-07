splits5_mitigated_lodo_lyria: leave-one-dataset-out split, InReDD-only training, with
position-aware sound negatives (built 2026-10-02 by tools/make_five_mitigated.py). Provenance:
  * source split: splits_lodo_lyria (untouched). test.csv is a verbatim copy (2,121
    expert-reviewed, non-frozen crops: 1,116 DENTEX + 1,005 Tufts).
  * train_pool.csv: splits_lodo_lyria/train_pool.csv (lyria only, 180 per class, implant capped at 93)
    with the 180 sound crops replaced by 60 third + 60 second molars + 60 other sound teeth from the
    lyria patients of that pool (make_posneg_pool algorithm, random.Random(0)).
  * folds.json: StratifiedGroupKFold(5, shuffle=True, random_state=0) (only test mode is trained,
    as for the published lodo_* runs).
  * crops: variants/five_mitigated/enhanced/none (x1.3 context, external crops re-extracted
    from their source radiograph with the recorded segmenter box).
