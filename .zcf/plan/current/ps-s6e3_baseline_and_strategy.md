# PS-S6E3 Baseline And Strategy

- Status: `approved -> executing`
- Task: 分析研究目的、构思冲榜策略、搭建可提交 baseline，并给出优化方向
- Constraint: 模型训练仅在 Kaggle 远程进行；本地仅做最小 smoke test
- Execution Started At: `2026-03-18 12:01:27 +08:00`
- Timestamp note: 当前环境无 `bash` 命令，时间由 PowerShell `Get-Date` 获取

## Context

1. Competition: `playground-series-s6e3`
2. Metric: `ROC AUC`
3. Data:
   - `train.csv` shape: `(594194, 21)`
   - `test.csv` shape: `(254655, 20)`
4. Public-solution patterns:
   - XGB + leak-free target encoding + pseudo label
   - Ridge -> XGB two-stage
   - rank/weight blend as final boost

## Approved Steps

1. 产出研究文档 `docs/competition_research.md`
2. 产出远程训练脚本 `src/remote/train_baseline_xgb_te.py`
3. 产出配置文件 `src/remote/config_baseline.json`
4. 产出本地 smoke 测试脚本 `src/local/smoke_test_pipeline.py`
5. 产出提交校验脚本 `src/local/validate_submission.py`
6. 产出融合脚本 `src/remote/blend_rank_average.py`
7. 产出优化路线 `docs/optimization_roadmap.md`
8. 更新 `README.md`

## Definition Of Done

1. 可在 Kaggle 远程直接训练并生成 `submission.csv`
2. 本地 smoke test 可短时跑通且产出格式正确
3. 提供可执行优化路线与冲榜策略
