# PS-S6E3 Baseline Workflow

本项目用于 `Kaggle Playground Series - Season 6 Episode 3`（Customer Churn）实战流程：
- 远程训练在 Kaggle 执行
- 本地只做最小 smoke test 和文件校验

## 当前状态

- 当前最佳 Public LB：`0.91606`
- 最佳提交：`phase9 realmlp low-weight blend v1`
- 最佳提交文件：`kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/submission_blend_opt.csv`
- 最强单模：`phase8 catboost strong v1` -> `0.91591`
- 当前最佳融合核心：`phase8_cat + phase6_cat + phase6_lgbm + phase3 + phase2 + low-weight RealMLP`

推荐优先阅读：

- `docs/competition_paper.md`
- `docs/competition_research.md`
- `docs/optimization_roadmap.md`

## 目录结构

```
PS-S6E3/
├── src/
│   ├── remote/
│   │   ├── train_baseline_xgb_te.py
│   │   ├── config_baseline.json
│   │   └── blend_rank_average.py
│   └── local/
│       ├── smoke_test_pipeline.py
│       └── validate_submission.py
├── docs/
│   ├── competition_research.md
│   └── optimization_roadmap.md
└── .zcf/plan/
    ├── current/
    └── history/
```

## 1) 本地 smoke test（仅验证链路）

```bash
python src/local/smoke_test_pipeline.py \
  --train-path train.csv \
  --test-path test.csv \
  --sample-submission-path sample_submission.csv \
  --max-train-rows 4000 \
  --max-test-rows 1500
```

输出目录：`.artifacts/smoke/output/`

## 2) Kaggle 远程训练 baseline

```bash
python src/remote/train_baseline_xgb_te.py \
  --train-path /kaggle/input/competitions/playground-series-s6e3/train.csv \
  --test-path /kaggle/input/competitions/playground-series-s6e3/test.csv \
  --sample-submission-path /kaggle/input/competitions/playground-series-s6e3/sample_submission.csv \
  --config-path src/remote/config_baseline.json \
  --output-dir /kaggle/working/outputs/baseline
```

关键产物：
- `submission.csv`
- `oof_predictions.csv`
- `cv_metrics.json`

## 3) 提交文件校验

```bash
python src/local/validate_submission.py \
  --submission-path outputs/baseline/submission.csv \
  --sample-submission-path sample_submission.csv
```

## 4) 多提交融合

```bash
python src/remote/blend_rank_average.py \
  --sample-submission-path sample_submission.csv \
  --submission-paths outputs/a.csv outputs/b.csv \
  --method weighted_rank \
  --weights 0.6 0.4 \
  --output-path outputs/submission_blend.csv
```

## 5) 调优建议

查看 `docs/optimization_roadmap.md`，按 P1 -> P2 -> P3 逐步推进。

## 6) 新增冲分脚本（Phase-5/6/7）

```bash
# Phase-5: XGB Advanced
python kaggle_kernel/phase5_xgb_advanced/train_xgb_advanced.py \
  --train-path /kaggle/input/competitions/playground-series-s6e3/train.csv \
  --test-path /kaggle/input/competitions/playground-series-s6e3/test.csv \
  --sample-submission-path /kaggle/input/competitions/playground-series-s6e3/sample_submission.csv \
  --config-path kaggle_kernel/phase5_xgb_advanced/config_xgb_advanced.json \
  --output-dir /kaggle/working/phase5/output_v1

# Phase-6: Diverse Tree (LGBM + CatBoost)
python kaggle_kernel/phase6_diverse_tree/train_lgbm_cat_diverse.py \
  --train-path /kaggle/input/competitions/playground-series-s6e3/train.csv \
  --test-path /kaggle/input/competitions/playground-series-s6e3/test.csv \
  --sample-submission-path /kaggle/input/competitions/playground-series-s6e3/sample_submission.csv \
  --config-path kaggle_kernel/phase6_diverse_tree/config_diverse_tree.json \
  --output-dir /kaggle/working/phase6/output_v1

# Phase-7: OOF constrained blending
python kaggle_kernel/phase7_blend_oof/blend_rank_oof_search.py \
  --config-path kaggle_kernel/phase7_blend_oof/blend_config.json \
  --sample-submission-path /kaggle/input/competitions/playground-series-s6e3/sample_submission.csv \
  --output-dir /kaggle/working/phase7/output_v1
```

本地快速链路测试：

```bash
python scripts/smoke/smoke_phase5_phase6.py \
  --train-path train.csv \
  --test-path test.csv \
  --sample-submission-path sample_submission.csv
```
