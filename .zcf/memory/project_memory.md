# PS-S6E3 Project Memory (Cross-Device Handoff)

Last Updated: 2026-03-23 11:39:50 +0800
Updated At (ISO): 2026-03-23T11:39:50+0800

## 1. Current Objective
- Competition: `playground-series-s6e3` (Predict Customer Churn)
- Goal: improve public LB beyond current best `0.91608`
- Constraint: training and submission must run on Kaggle remote; local only handles smoke tests, OOF analysis, config management, and documentation

## 2. Best Known Results
- Best public LB: `0.91608`
  - submission: `phase10 stack oof v1`
  - interpretation: first stacking route that beat the previous `phase9` blend online
- Best local OOF: `0.9184734996`
  - route: `phase14 stronger stack pipeline v1/v2`
  - interpretation: strongest local second-layer pipeline, but not best on Public LB
- Strongest single model:
  - `phase8 catboost strong v1`
  - OOF AUC: `0.9181653`
  - Public LB: `0.91591`
- Strongest pre-stack blend:
  - `phase9 realmlp low-weight blend v1`
  - blended OOF AUC: `0.9184601`
  - Public LB: `0.91606`

## 3. Latest Submission Ladder
- `phase10 stack oof v1` -> `0.91608` (current best public)
- `phase14 stronger stack pipeline v1` -> `0.91607`
- `phase12 rank hybrid v1` -> `0.91607`
- `phase11 stack blend hybrid v1` -> `0.91607`
- `phase9 realmlp low-weight blend v1` -> `0.91606`
- `phase7 phase8 candidate blend opt v1` -> `0.91602`
- `phase8 catboost strong v1` -> `0.91591`
- `phase7 phase6 candidate blend opt v1` -> `0.91591`
- `phase6 catboost v1` -> `0.91581`
- `phase6 ensemble v1` -> `0.91567`
- `phase5 xgb advanced v1` -> `0.89306` (dead route)

## 4. Latest Route Summary

### 4.1 Phase10
- Folder: `kaggle_kernel/phase10_stack_oof`
- Purpose: build second-layer stacking on top of existing OOF/submission files
- Best candidate:
  - `candidate_set=all_core`
  - `feature_mode=raw_rank_logit`
  - `meta_model=logreg_l2_c0p25`
- Result:
  - reference blend OOF: `0.9184677453`
  - stack best OOF: `0.9184538058`
  - Public LB: `0.91608`
- Takeaway:
  - even though OOF was slightly lower than the best phase9 blend, stacking improved online score
  - this is the strongest evidence that second-layer stacking is a real route, not just local overfit

### 4.2 Phase11 and Phase12
- Folders:
  - `kaggle_kernel/phase11_stack_blend_hybrid`
  - `kaggle_kernel/phase12_rank_hybrid`
- Purpose:
  - do narrow hybrid search between `phase10 stack best` and `phase9 reference blend`
- Best evidence:
  - local best sits around `rank` space with `stack_weight ~= 0.17`
  - local OOF rises to about `0.9184687`
- Public LB:
  - both routes only reached `0.91607`
- Takeaway:
  - hybrid route is valid
  - but weight-only refinement is already near platform on the current model pool

### 4.3 Phase13
- Folder: `kaggle_kernel/phase13_hybrid_plus_realmlp`
- Purpose:
  - add `phase9_realmlp` back as a very small low-correlation correction on top of `phase12 best`
- Best local evidence:
  - best mode: `rank`
  - best `realmlp_weight ~= 0.02`
  - local OOF around `0.9184704303`
- Takeaway:
  - `RealMLP` still works better as a weak auxiliary correction than as a standalone main branch
  - no new public best was established from this route

### 4.4 Phase14
- Folder: `kaggle_kernel/phase14_stronger_stack_pipeline`
- Purpose:
  - move from simple second-layer probability columns to a richer stacking feature pipeline
- Inputs:
  - `phase13_hybrid_best_v1`
  - `phase10_stack_best_v1`
  - `phase7_blend_best_v1`
  - `phase8_cat_v1`
  - `phase9_realmlp_v2`
- Feature packs:
  - `raw`
  - `rank`
  - `logit`
  - stats features
  - anchor-gap features
  - pairwise absdiff features
- Best candidate:
  - `stack_plus_diversity + full_linear + logreg_newtoncg_c0p25`
- Result:
  - OOF AUC: `0.9184734996`
  - Public LB: `0.91607`
- Takeaway:
  - current strongest local route
  - but not stronger online than `phase10`
  - the current bottleneck is likely base-model diversity, not second-layer feature engineering

## 5. Implemented Assets

### 5.1 Kaggle kernels
- `kaggle_kernel/phase10_stack_oof`
- `kaggle_kernel/phase11_stack_blend_hybrid`
- `kaggle_kernel/phase12_rank_hybrid`
- `kaggle_kernel/phase13_hybrid_plus_realmlp`
- `kaggle_kernel/phase14_stronger_stack_pipeline`

### 5.2 Kaggle datasets
- `kaggle_dataset/phase10_stack_inputs`
- `kaggle_dataset/phase11_hybrid_inputs`
- `kaggle_dataset/phase13_realmlp_inputs`
- `kaggle_dataset/phase14_stack_pipeline_inputs`

### 5.3 Supporting files
- `scripts/smoke/smoke_phase10_stack_oof.py`
- `scripts/smoke/smoke_phase11_stack_blend_hybrid.py`
- `scripts/smoke/smoke_phase12_rank_hybrid.py`
- `scripts/smoke/smoke_phase13_hybrid_plus_realmlp.py`
- `scripts/smoke/smoke_phase14_stronger_stack_pipeline.py`

### 5.4 Documentation
- `docs/competition_paper.md`
- `README.md`
- `.zcf/plan/current/ps-s6e3_model_optimization.md`
- `.zcf/plan/history/2026-03-22_125517_phase10_stack_oof_v1.md`
- `.zcf/plan/history/2026-03-22_131417_phase11_stack_blend_hybrid_v1.md`
- `.zcf/plan/history/2026-03-22_132431_phase12_rank_hybrid_v1.md`
- `.zcf/plan/history/2026-03-22_134457_phase13_hybrid_plus_realmlp_v1.md`
- `.zcf/plan/history/2026-03-22_195235_phase14_stronger_stack_pipeline_v1.md`

## 6. Current Recommended Route
- Keep `phase10 stack oof v1` as the current online anchor.
- Keep `phase14 stronger stack pipeline` as the current local research framework.
- Do not spend more rounds on pure `phase11-14` weight squeezing with the same base model pool.
- Next high-value action:
  - add a truly new low-correlation base model or stronger feature-enhanced branch
  - then feed it into `phase14`

## 7. Dead / Low-Value Routes
- `phase5_xgb_advanced`
  - fixed dataset mounting but still weak after repair
  - do not submit again
- standalone `phase9_realmlp_v2`
  - useful only as a weak auxiliary blend input
  - not strong enough as an independent primary submission

## 8. Resume Checklist
1. Read `docs/competition_paper.md` for the latest beginner-friendly full summary.
2. Use `phase10 stack oof v1` as the default public benchmark to beat.
3. Use `phase14 stronger stack pipeline` if a new base model needs to be tested in a stronger second layer.
4. Before any new submission, re-check leaderboard history:
   - `kaggle competitions submissions -c playground-series-s6e3 -v`
5. Prefer:
   - new diverse base models
   - stronger original-signal feature branches
   - cleaner OOF inputs for phase14
6. Avoid:
   - more phase5 work
   - public-LB-only weight tuning on the existing phase10-14 pool

## 9. Operational Notes
- Local machine is still for smoke tests and documentation only.
- Kaggle remote remains the only place for full search, training, and official submission.
- Current repository now contains phase10-14 code, supporting Kaggle datasets, and updated handoff documents.
