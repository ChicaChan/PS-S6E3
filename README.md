# PS-S6E3 Competition Workflow

本项目用于 `Kaggle Playground Series - Season 6 Episode 3`（Customer Churn）实战流程：
- 正式训练与提交全部在 Kaggle 远程执行
- 本地只做最小 smoke test、配置维护、OOF 分析与文档归档

## 当前状态

- 当前最佳 Public LB：`0.91608`
- 最佳线上提交：`phase10 stack oof v1`
- 当前最佳本地 OOF：`phase14 stronger stack pipeline v1/v2` -> `0.9184734996`
- 当前最强单模：`phase8 catboost strong v1` -> `0.91591`
- 当前研究结论：
  - `phase10` 证明二层 stacking 已在线上产生真实增益
  - `phase11-14` 继续微调二层融合，只抬高本地 OOF，没有超过 `0.91608`
  - 下一轮优先级应转向“新增低相关底模”，再复用 `phase14` 强二层管线

推荐优先阅读：

- `docs/competition_paper.md`
- `.zcf/plan/current/ps-s6e3_model_optimization.md`
- `.zcf/plan/history/2026-03-22_195235_phase14_stronger_stack_pipeline_v1.md`

## 目录结构

```text
PS-S6E3/
├── docs/
│   ├── competition_paper.md
│   ├── competition_research.md
│   ├── optimization_roadmap.md
│   └── powerbi_eda_beginner_guide.md
├── kaggle_kernel/
│   ├── phase8_catboost_strong/
│   ├── phase9_realmlp_tabm_diverse/
│   ├── phase10_stack_oof/
│   ├── phase11_stack_blend_hybrid/
│   ├── phase12_rank_hybrid/
│   ├── phase13_hybrid_plus_realmlp/
│   └── phase14_stronger_stack_pipeline/
├── kaggle_dataset/
│   ├── phase10_stack_inputs/
│   ├── phase11_hybrid_inputs/
│   ├── phase13_realmlp_inputs/
│   └── phase14_stack_pipeline_inputs/
├── scripts/
│   └── smoke/
└── .zcf/
    ├── memory/
    └── plan/
```

## 1) 本地 Smoke Test

基线最小链路验证：

```bash
python src/local/smoke_test_pipeline.py \
  --train-path train.csv \
  --test-path test.csv \
  --sample-submission-path sample_submission.csv \
  --max-train-rows 4000 \
  --max-test-rows 1500
```

输出目录：`.artifacts/smoke/output/`

## 2) Kaggle 远程训练 Baseline

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

## 4) 第一轮有效冲榜主线

- `phase6_diverse_tree/`
  - `CatBoost` / `LightGBM` 多样性树模型
- `phase7_blend_oof/`
  - 基于 OOF 相关性过滤的受限融合
- `phase8_catboost_strong/`
  - 强特征版 `CatBoost`
- `phase9_realmlp_tabm_diverse/`
  - `RealMLP` 低权重多样性补丁

这一阶段的最佳线上结果：
- `phase9 realmlp low-weight blend v1` -> `0.91606`

## 5) 第二轮 Stacking / Hybrid 主线

- `phase10_stack_oof/`
  - 首次把 `stacking` 真正转化为线上增益
  - 当前最佳线上提交来自这一阶段：`0.91608`
- `phase11_stack_blend_hybrid/`
  - `phase10 stack best` 与 `phase9 reference blend` 的二元 hybrid
- `phase12_rank_hybrid/`
  - 窄区间 `rank` 权重微调
- `phase13_hybrid_plus_realmlp/`
  - 在 `phase12 best` 上叠加 `RealMLP` 小权重增强
- `phase14_stronger_stack_pipeline/`
  - 更强的二层特征管线：`raw/rank/logit + stats + anchor gaps + pairwise absdiff`

当前阶段性结论：
- `phase10` 是当前线上最好解
- `phase14` 是当前本地 OOF 最强解
- 但 `phase14` 没能在线上超过 `phase10`

## 6) Phase10-14 本地验证命令

```bash
python scripts/smoke/smoke_phase10_stack_oof.py \
  --train-path train.csv \
  --test-path test.csv \
  --sample-submission-path sample_submission.csv

python scripts/smoke/smoke_phase14_stronger_stack_pipeline.py \
  --train-path train.csv \
  --test-path test.csv \
  --sample-submission-path sample_submission.csv
```

说明：
- 本地 smoke 只验证脚本链路
- 完整搜索、正式提交和排行榜计分全部在 Kaggle 远程完成

## 7) 当前推荐策略

1. 把 `phase10 stack oof v1` 作为当前线上稳定锚点。
2. 把 `phase14 stronger stack pipeline` 作为后续实验框架继续复用。
3. 下一轮优先新增低相关底模，而不是继续微调 `phase11-14` 的权重区间。

## 8) 文档维护提示

- `README.md` 记录当前最优线上路线和主要目录
- `docs/competition_paper.md` 记录完整比赛方案与教学型分析
- `.zcf/memory/project_memory.md` 作为跨设备 handoff 记忆
