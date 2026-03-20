# PS-S6E3 Project Memory (Cross-Device Handoff)

Last Updated: 2026-03-20 15:14:43 +0800
Updated At (ISO): 2026-03-20T15:14:43+0800

## 1. Current Objective
- Competition: `playground-series-s6e3` (Predict Customer Churn)
- Goal: improve public LB beyond current best `0.91606`
- Constraint: training/submission only on Kaggle remote; local only smoke/validation

## 2. Best Known Results
- Historical best public LB: `0.91606`
  - `phase9 realmlp low-weight blend v1` -> 0.91606
  - previous bests:
    - `phase7 phase8 candidate blend opt v1` -> 0.91602
    - `phase8 catboost strong v1` -> 0.91591
    - `phase7 phase6 candidate blend opt v1` -> 0.91591
    - `phase6 catboost v1` -> 0.91581
    - `phase4 blend eq rank` / `phase4 blend opt rank` -> 0.91407

Recent submissions (latest run):
- `phase9 realmlp low-weight blend v1` -> `0.91606` (new best)
- `phase7 phase8 candidate blend opt v1` -> `0.91602` (new best)
- `phase8 catboost strong v1` -> `0.91591` (ties current best)
- `phase7 phase6 candidate blend opt v1` -> `0.91591` (new best)
- `phase6 ensemble v1` -> `0.91567`
- `phase6 catboost v1` -> `0.91581` (new best)
- `phase5 xgb advanced v1` -> `0.89306` (regression)
- `phase7 blend opt now` -> `0.90109` (regression)

## 3. Root Cause of Regression
- `phase5 v1` ran without original Telco dataset mounted.
- Script fallback used train as reference distribution:
  - leads to leakage-like overfit and poor public LB generalization.
- Fix applied: add dataset sources in
  - `kaggle_kernel/phase5_xgb_advanced/kernel-metadata.json`
  - datasets: `blastchar/telco-customer-churn`, `cdeotte/s6e3-original-dataset`

## 4. Remote Runtime Status (latest checked)
- `chicachan/ps-s6e3-xgb-advanced-v1`: `COMPLETE`
- `chicachan/ps-s6e3-diverse-tree-v1`: `COMPLETE`
- `chicachan/ps-s6e3-realmlp-tabm-diverse-v1`: `COMPLETE`
- `phase7` remains a local-only blend workflow because Kaggle notebook source mounting failed in the earlier attempt.

## 5. New Implemented Assets
- Documentation:
  - `docs/competition_paper.md`
  - `.zcf/plan/history/2026-03-20_151709_ps-s6e3_round2_diversity_push.md`
- Phase-8:
  - `kaggle_kernel/phase8_catboost_strong/train_catboost_strong.py`
  - `kaggle_kernel/phase8_catboost_strong/config_catboost_strong.json`
  - `kaggle_kernel/phase8_catboost_strong/kernel-metadata.json`
  - `scripts/smoke/smoke_phase8_catboost_strong.py`
- Phase-9:
  - `kaggle_kernel/phase9_realmlp_tabm_diverse/train_realmlp_tabm_diverse.py`
  - `kaggle_kernel/phase9_realmlp_tabm_diverse/config_realmlp_tabm_diverse.json`
  - `kaggle_kernel/phase9_realmlp_tabm_diverse/kernel-metadata.json`
  - `scripts/smoke/smoke_phase9_realmlp_tabm_diverse.py`
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

## 6. Key New Findings
- `phase5_xgb_advanced/output_v2/cv_metrics.json`
  - OOF AUC: `0.909806`
  - log confirms original reference data was mounted
  - conclusion: route is intrinsically weak; stop investing in phase5
- `phase6_diverse_tree/output_v1/cv_metrics_lgbm.json`
  - LGBM OOF AUC: `0.9158487`
- `phase6_diverse_tree/output_v1/cv_metrics_cat.json`
  - CatBoost OOF AUC: `0.9180636`
- `phase6_diverse_tree/output_v1/phase6_report.json`
  - ensemble OOF AUC: `0.9179283`
- Local phase7 candidate search on `phase6_cat + phase6_lgbm + phase2 + phase3 + th999`
  - config: `kaggle_kernel/phase7_blend_oof/local_blend_config_phase6_candidates.json`
  - correlation threshold: `0.994`
  - selected models: `phase6_cat_v1`, `phase2_fe_v1`, `phase6_lgbm_v1`
  - best method: `prob`
  - best OOF AUC: `0.9183020`
  - equal-rank OOF AUC: `0.9180864`
  - output dir: `kaggle_kernel/phase7_blend_oof/output_phase6_candidates`
- `phase8_catboost_strong/output_v2/cv_metrics_cat.json`
  - CatBoost Strong OOF AUC: `0.9181653`
  - feature count: `119`
  - created original signals:
    - `orig_single`: `41`
    - `orig_cross`: `15`
    - `dist`: `8`
  - conclusion: better than `phase6_cat`, but only by a narrow margin
- Local phase7 candidate search on `phase8_cat + phase6_cat + phase6_lgbm + phase2 + phase3`
  - config: `kaggle_kernel/phase7_blend_oof/local_blend_config_phase8_candidates.json`
  - correlation threshold: `0.999`
  - best method: `prob`
  - best OOF AUC: `0.9183841`
  - equal-rank OOF AUC: `0.9181751`
  - output dir: `kaggle_kernel/phase7_blend_oof/output_phase8_candidates`
  - note: `phase8_cat` and `phase6_cat` correlation is very high (`0.998622`), so this is still a same-family enhancement, not true diversity
- `phase9_realmlp_tabm_diverse` scaffold is ready
  - route: `RealMLP_TD_Classifier + TabM_D_Classifier` via `pytabkit`
  - feature backbone reused from `phase8` strong-feature chain
  - Kaggle kernel id:
    - `chicachan/ps-s6e3-realmlp-tabm-diverse-v1`
  - metadata:
    - `enable_gpu: true`
    - `enable_internet: true`
  - objective:
    - validate whether a lower-correlation neural/tabular family can improve final blend beyond `0.91602`
  - push status:
    - version `v2` has completed on Kaggle
    - output files have been downloaded to:
      - `kaggle_kernel/phase9_realmlp_tabm_diverse/output_v2`
  - cancellation status:
    - current local official Kaggle CLI / SDK does not expose a public cancel interface
    - direct probe of likely internal stop endpoints returned `404`
    - practical meaning: `v1` cannot be programmatically cancelled from the currently available local tooling
  - local code has now been tightened into `v2` shape:
    - `n_folds=3`
    - `inner_folds=3`
    - `enable_tabm=false`
    - `RealMLP` only for the first fast diversity check
    - explicit `n_epochs/batch_size/hidden_sizes`
    - fold-level heartbeat logging added
  - v2 actual result:
    - `RealMLP` OOF AUC: `0.9147087`
    - correlation vs `phase6_cat`: `0.980665`
    - correlation vs `phase8_cat`: `0.980994`
    - verdict:
      - route is runnable and moderately diverse
      - but model strength is not enough for direct submission
  - low-weight blend result:
    - best blend method used locally:
      - inject `RealMLP` into current best phase8 candidate blend by 1-D prob grid
    - best `RealMLP` weight:
      - `0.124`
    - blended OOF AUC:
      - `0.9184601`
    - previous best blend OOF:
      - `0.9183841`
    - delta:
      - `+0.0000761`
    - verdict:
      - small but real positive gain
      - this candidate is worth one leaderboard submission test
  - leaderboard result:
    - submission message:
      - `phase9 realmlp low-weight blend v1`
    - Public LB:
      - `0.91606`
    - final verdict:
      - current best public solution

## 7. Local Validation Status
- Smoke test passed for chain skeleton.
- On current local machine, `lightgbm/catboost` not installed; phase6 smoke auto-skip is expected.
- `phase8` static compile passed.
- `phase8` smoke runner executed successfully and skipped runtime training because local `catboost` is not installed.
- `phase9` static compile passed.
- `phase9` smoke runner executed successfully and skipped runtime training because local `pytabkit` is not installed.
- Submission format validated locally for:
  - `phase6_diverse_tree/output_v1/submission_cat.csv`
  - `phase6_diverse_tree/output_v1/submission.csv`
  - `phase7_blend_oof/output_phase6_candidates/submission_blend_eq.csv`
  - `phase7_blend_oof/output_phase6_candidates/submission_blend_opt.csv`
  - `phase8_catboost_strong/output_v2/submission.csv`
  - `phase8_catboost_strong/output_v2/submission_cat.csv`
  - `phase7_blend_oof/output_phase8_candidates/submission_blend_opt.csv`

## 8. Resume Checklist (Next Device / Next Session)
1. Primary submission candidate:
   - already validated and submitted:
   - `kaggle_kernel/phase6_diverse_tree/output_v1/submission_cat.csv`
   - result: `0.91581`
2. Secondary submission candidate:
   - already validated and submitted:
   - `kaggle_kernel/phase6_diverse_tree/output_v1/submission.csv`
   - result: `0.91567`
3. Blend candidate worth testing after single-model sanity check:
   - already validated and submitted:
   - `kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/submission_blend_opt.csv`
   - result: `0.91606`
   - this is the current best public submission
4. Current practical objective:
   - keep current best blend as stable baseline
   - next route should aim for another genuinely complementary model family or stronger second-layer stack
   - current best submission is:
   - `phase7_blend_oof/output_phase9_realmlp_candidates/submission_blend_opt.csv`
5. Before any new submission, re-check recent leaderboard responses:
   - `kaggle competitions submissions -c playground-series-s6e3 -v`
6. Keep avoiding:
   - any `phase5` submission
   - any blend dominated by `phase5` predictions
7. Round-2 code scaffold is ready:
   - Kaggle kernel folder: `kaggle_kernel/phase8_catboost_strong`
   - suggested push target: `chicachan/ps-s6e3-catboost-strong-v1`
8. New diversity scaffold is also ready:
   - Kaggle kernel folder: `kaggle_kernel/phase9_realmlp_tabm_diverse`
   - suggested push target: `chicachan/ps-s6e3-realmlp-tabm-diverse-v1`
   - `v2` push and remote run already completed
   - best next action:
   - use `RealMLP` only as a low-weight blend candidate, not as standalone submit
   - current best local candidate file:
   - `kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/submission_blend_opt.csv`

## 9. Key Notes
- Keep pseudo-label gating enabled (only accept fold-level improvement).
- Do not submit fallback-reference models (train-as-reference) again.
- Prefer OOF-constrained blend over public-only heuristic weighting.
- Current highest-value route is `phase9` diversity model first, then evaluate whether it improves Track-C blend over the current `0.91602` best.
- `phase9 v2` has now answered the diversity question:
  - keep only as a weak auxiliary blend candidate unless later low-weight blend shows real gain
- latest answer:
  - low-weight blend already showed a small real gain, so this route remains alive only in ensemble form
- latest leaderboard answer:
  - this low-weight blend has already improved Public LB to `0.91606`
