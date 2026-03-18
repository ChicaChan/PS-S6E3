# PS-S6E3 Project Memory (Cross-Device Handoff)

Last Updated: 2026-03-18 18:04:39 (Asia/Shanghai)
Updated At (ISO): 2026-03-18T18:04:39+08:00

## 1. Current Objective
- Competition: `playground-series-s6e3` (Predict Customer Churn)
- Goal: improve public LB beyond historical best `0.91407`
- Constraint: training/submission only on Kaggle remote; local only smoke/validation

## 2. Best Known Results
- Historical best public LB: `0.91407`
  - `phase4 blend eq rank` -> 0.91407
  - `phase4 blend opt rank` -> 0.91407

Recent submissions (latest run):
- `phase5 xgb advanced v1` -> `0.89306` (regression)
- `phase7 blend opt now` -> `0.90109` (regression)

## 3. Root Cause of Regression
- `phase5 v1` ran without original Telco dataset mounted.
- Script fallback used train as reference distribution:
  - leads to leakage-like overfit and poor public LB generalization.
- Fix applied: add dataset sources in
  - `kaggle_kernel/phase5_xgb_advanced/kernel-metadata.json`
  - datasets: `blastchar/telco-customer-churn`, `cdeotte/s6e3-original-dataset`

## 4. Remote Runtime Status (at update time)
- `chicachan/ps-s6e3-xgb-advanced-v1`: `QUEUED` (v2 pushed with dataset fix)
- `chicachan/ps-s6e3-diverse-tree-v1`: `RUNNING`
- `chicachan/ps-s6e3-blend-oof-search-v1`: pushed, but kernel source mounting failed
  - practical workaround: run blend locally after downloading remote outputs

## 5. New Implemented Assets
- Phase-5:
  - `kaggle_kernel/phase5_xgb_advanced/train_xgb_advanced.py`
  - `kaggle_kernel/phase5_xgb_advanced/config_xgb_advanced.json`
  - `kaggle_kernel/phase5_xgb_advanced/kernel-metadata.json`
- Phase-6:
  - `kaggle_kernel/phase6_diverse_tree/train_lgbm_cat_diverse.py`
  - `kaggle_kernel/phase6_diverse_tree/config_diverse_tree.json`
  - `kaggle_kernel/phase6_diverse_tree/kernel-metadata.json`
- Phase-7:
  - `kaggle_kernel/phase7_blend_oof/blend_rank_oof_search.py`
  - `kaggle_kernel/phase7_blend_oof/blend_config.json`
  - `kaggle_kernel/phase7_blend_oof/kernel-metadata.json`
- Local smoke:
  - `scripts/smoke/smoke_phase5_phase6.py`

## 6. Local Validation Status
- Smoke test passed for chain skeleton.
- On current local machine, `lightgbm/catboost` not installed; phase6 smoke auto-skip is expected.

## 7. Resume Checklist (Next Device / Next Session)
1. Check remote statuses:
   - `kaggle kernels status chicachan/ps-s6e3-xgb-advanced-v1`
   - `kaggle kernels status chicachan/ps-s6e3-diverse-tree-v1`
2. Download outputs when complete:
   - `kaggle kernels output chicachan/ps-s6e3-xgb-advanced-v1 -p kaggle_kernel/phase5_xgb_advanced/output_v2`
   - `kaggle kernels output chicachan/ps-s6e3-diverse-tree-v1 -p kaggle_kernel/phase6_diverse_tree/output_v1`
3. Create local blend config using downloaded OOF + submissions, then run:
   - `python kaggle_kernel/phase7_blend_oof/blend_rank_oof_search.py --config-path <local_config> --sample-submission-path sample_submission.csv --output-dir kaggle_kernel/phase7_blend_oof/output_next`
4. Validate submission format:
   - `python src/local/validate_submission.py --submission-path <blend_file> --sample-submission-path sample_submission.csv`
5. Submit and compare with `0.91407` baseline.

## 8. Key Notes
- Keep pseudo-label gating enabled (only accept fold-level improvement).
- Do not submit fallback-reference models (train-as-reference) again.
- Prefer OOF-constrained blend over public-only heuristic weighting.
