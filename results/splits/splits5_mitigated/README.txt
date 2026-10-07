splits5_mitigated: five-class split with position-aware sound negatives (built 2026-10-02 by
tools/make_five_mitigated.py). Provenance:
  * source split: splits_clean5 (restored-class partition; untouched). test.csv is a
    verbatim copy of splits_clean5/test.csv (3,017 crops: 914 endo, 983 sound, 133 impacted, 18 implant,
    969 restored), so every test patient keeps its side.
  * train_pool.csv: splits_clean5/train_pool.csv with its 180 sound crops replaced by 60 third molars,
    60 second molars and 60 other sound teeth (strict state H), drawn round-robin patient-diverse with
    random.Random(0) from the lyria patients of that same training pool (algorithm of
    tools/make_variants.py make_posneg_pool). Every other row (endo, impacted, implant incl. the
    expert-reviewed Tufts/DENTEX implant fill, restored) is unchanged.
  * folds.json: StratifiedGroupKFold(5, shuffle=True, random_state=0) on the new pool, groups = patient
    (lyria) or radiograph (external), as in make_variants.py and make_splits_clean.py.
  * crops are read by train.py from variants/five_mitigated/enhanced/none/<Class>/<basename>,
    which holds every crop re-extracted with its box scaled x1.3 about the centre (IEI_ENHANCED).
Summary: see split_summary.json and variants/five_mitigated/build_log.json.
