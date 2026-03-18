# PS-S6E3 研究结论

## 1. 赛题目标

- 比赛：`playground-series-s6e3`
- 任务：二分类，预测 `Churn`
- 指标：`ROC AUC`

## 2. 数据与约束

- 训练集：`594,194` 行，`21` 列（含目标列）
- 测试集：`254,655` 行，`20` 列
- 约束：模型训练必须在 Kaggle 远程执行；本地只做最小可运行检查

## 3. 公共方案调研要点（截至 2026-03-18）

1. 单模主线稳定有效：
   - `XGBoost + leak-free target encoding + optional pseudo label`
2. 进阶冲榜常见：
   - `Ridge -> XGB` 两阶段
   - 类别组合特征（bi-gram/tri-gram）
3. 榜单后期增益常见：
   - 多提交融合（均值、rank 融合、权重融合、hill climbing）

## 4. 本项目策略落点

1. 第一优先级：先产出可提交 baseline（稳定、可复现、便于调参）
2. 第二优先级：基于 baseline 扩展冲榜（两阶段、特征增强）
3. 第三优先级：多模型融合（作为后期 LB 提升工具）

## 5. 风险与规避

1. 泄漏风险：
   - TE 必须在 outer fold 内部完成 inner-fold OOF 编码
2. 伪标签风险：
   - 只使用高置信样本并和 base fold AUC 比较，未提升则回退
3. Public LB 过拟合风险：
   - 融合权重以 OOF 或多次提交稳定性为依据，不追单次波动

## 6. 当前交付范围

1. 远程训练脚本：`src/remote/train_baseline_xgb_te.py`
2. 基线配置：`src/remote/config_baseline.json`
3. 本地最小跑通：`src/local/smoke_test_pipeline.py`
4. 提交校验：`src/local/validate_submission.py`
5. 融合脚本：`src/remote/blend_rank_average.py`
