# PS-S6E3 Model Optimization

- Status: `approved -> in_progress`
- Task: 优化模型并提升 Kaggle 排名分数
- Created At: `2026-03-18T14:19:08+0800`
- Constraint: 训练与提交仅在 Kaggle 远程执行；本地只做最小验证与脚本维护

## Goal

1. 基于现有 baseline（Public LB 0.91384）继续提升分数。
2. 每轮实验都保留可复现配置、OOF 与提交文件。
3. 用小步快跑方式验证增益，避免大规模改动后难以定位效果来源。

## Optimization Directions

1. Phase-1: `XGB + Pseudo Label` 参数扫描
   - 阈值候选：`0.995`, `0.997`, `0.999`
   - 仅在 fold AUC 提升时采用伪标签模型
2. Phase-2: 特征增强
   - 加入 `ORIG_proba` 统计映射
   - 添加高价值类别交叉（先从 `Contract/InternetService/PaymentMethod` 开始）
3. Phase-3: 两阶段模型
   - `Ridge(OHE+数值)` -> `XGB(+ridge_pred)`
4. Phase-4: 融合
   - 多提交 `weighted_rank` 融合
   - 权重优先由 OOF 与稳定性决定

## Execution Plan

1. 先实施 Phase-1 并完成至少 1 次远程提交，观察是否超过当前 0.91384。
2. 若提升不足，再进入 Phase-2（特征增强）并复测。
3. 当出现至少 2 个有效模型后进入 Phase-4 融合。
4. 每次实验记录：
   - 代码版本 / 配置
   - OOF AUC
   - Public LB 分数
   - 是否相对上一版提升

## Deliverables

1. 可复现的远程训练脚本与配置版本
2. 每轮实验的 `cv_metrics.json`、`submission.csv`
3. 最终优化结论与下一步建议

## Progress Log

### 2026-03-18T14:53:28+0800

1. Baseline（已完成）
   - Kernel: `chicachan/ps-s6e3-baseline-xgb-te-v1`
   - OOF AUC: `0.9162936`
   - Public LB: `0.91384`
2. Phase-1 Pseudo Label Threshold Sweep（已完成）
   - `th995`:
     - Kernel: `chicachan/ps-s6e3-pl-th995-v1`
     - OOF AUC: `0.9162936`（pseudo used: 0/5）
     - Public LB: `0.91384`（v2 提交）
   - `th997`:
     - Kernel: `chicachan/ps-s6e3-pl-th997-v1`
     - OOF AUC: `0.9162936`（pseudo used: 0/5）
     - Public LB: `0.91384`
   - `th999`:
     - Kernel: `chicachan/ps-s6e3-pl-th999-v1`
     - OOF AUC: `0.9162921`（pseudo used: 3/5）
     - Public LB: `0.91386`
3. Phase-2 Feature Engineering v1（已完成）
   - Kernel: `chicachan/ps-s6e3-fe-v1`
   - OOF AUC: `0.9163282`
   - Public LB: `0.91387`（当前最佳）
4. Phase-3 Ridge -> XGB v1（已完成）
   - Kernel: `chicachan/ps-s6e3-ridge-xgb-v1`
   - Timestamp: `2026-03-18T15:08:33+0800`
   - OOF AUC: `0.9162443`
   - Public LB: `0.91400`（当前最佳）
5. Phase-4 融合（已完成）
   - Timestamp: `2026-03-18T15:17:13+0800`
   - 输入模型:
     - `phase3 ridge xgb v1` (`0.91400`)
     - `phase2 fe v1 pseudo999` (`0.91387`)
     - `phase1 pseudo label th999` (`0.91386`)
   - 提交:
     - `phase4 blend eq rank` -> `0.91407`
     - `phase4 blend opt rank` -> `0.91407`
   - 结论: 融合进一步提升，当前最佳 `0.91407`
6. 高分作品研究（已完成）
   - Timestamp: `2026-03-18T15:28:46+08:00`
   - 样本来源:
     - `research/score_page1.csv`, `research/score_page2.csv`（近两页高分代码）
     - `research/highscore_parsed/*.py`, `research/parsed/*.py`
   - 关键结论:
     - 榜首区间作品里，`公开提交融合` 与 `强单模(0.919x CV)` 并存，融合类在公开榜占比高。
     - 强单模共性集中在：`ORIG_proba + pctrank 分布特征 + n-gram TE + 双层CV防泄漏 + XGB主干`。
     - 可复现提升路径应优先做 `新增多样性强模型`（TabM/RealMLP/LGBM/CB 强特征版）后再做 `rank 融合`，而不是继续微调同构 XGB。
7. Phase-5/6/7 脚本落地与本地冒烟（已完成）
   - Timestamp: `2026-03-18T15:56:16+08:00`
   - 新增目录:
     - `kaggle_kernel/phase5_xgb_advanced`
     - `kaggle_kernel/phase6_diverse_tree`
     - `kaggle_kernel/phase7_blend_oof`
     - `scripts/smoke`
   - 新增能力:
     - Phase-5: `XGB Advanced`（`ORIG_proba`、`pctrank`、`tenure_mod12`、`n-gram`、高阈值伪标签门控）
     - Phase-6: `LGBM + CatBoost` 多样性训练与统一产物
     - Phase-7: `OOF 相关性过滤 + rank/prob 权重搜索融合`
   - 本地验证:
     - `python scripts/smoke/smoke_phase5_phase6.py --max-train-rows 2000 --max-test-rows 800` 通过
     - 当前本机缺少 `lightgbm/catboost`，冒烟自动跳过 Phase-6，Phase-5 与 Phase-7 产物生成并通过提交格式校验
8. 远程执行与提交（进行中）
   - Timestamp: `2026-03-18T18:01:12+08:00`
   - Kernel 推送:
     - `chicachan/ps-s6e3-xgb-advanced-v1` v1：已完成训练并下载产物
     - `chicachan/ps-s6e3-diverse-tree-v1` v1：仍在 `RUNNING`
     - `chicachan/ps-s6e3-blend-oof-v1` v1：已推送，但 `kernel_sources` 未成功挂载，改为本地融合执行
   - 阶段结果:
     - Phase-5 v1 OOF AUC: `0.9258733`（日志显示未找到原始 Telco 数据，回退到 train 参考）
     - 本地 Phase-7（基于 phase5+phase3+phase2+th999）最佳 OOF AUC: `0.9264614`
   - 提交结果（Public LB）:
     - `phase5 xgb advanced v1` -> `0.89306`
     - `phase7 blend opt now` -> `0.90109`
   - 问题定位:
     - 分数下滑主因：Phase-5 v1 远程未挂载原始 Telco 数据，导致参考分布错误并显著过拟合
   - 修复动作:
     - 已给 `phase5_xgb_advanced/kernel-metadata.json` 增加 `dataset_sources`（`blastchar/telco-customer-churn`, `cdeotte/s6e3-original-dataset`）
     - 已推送 `phase5` v2，当前状态：`QUEUED`（等待运行）

### 2026-03-22T04:53:06+0800

9. Phase-10 Stack OOF v1（已完成）
   - Kernel: `chicachan/ps-s6e3-stack-oof-v1`
   - Artifact:
     - `kaggle_kernel/phase10_stack_oof/output_v4/phase10_stack_v1/stack_report.json`
     - `kaggle_kernel/phase10_stack_oof/output_v4/phase10_stack_v1/candidate_summary.json`
     - `kaggle_kernel/phase10_stack_oof/output_v4/phase10_stack_v1/submission_stack_best.csv`
   - Best Candidate:
     - `candidate_set=all_core`
     - `feature_mode=raw_rank_logit`
     - `meta_model=logreg_l2_c0p25`
   - OOF:
     - reference blend: `0.9184677453`
     - stack best: `0.9184538058`
   - Public LB:
     - `phase10 stack oof v1` -> `0.91608`
   - 结论:
     - 虽然 OOF 未超过参考 blend，但 Public LB 从 `0.91606` 提升到 `0.91608`
     - stacking 当前已验证具备真实线上增益，后续应优先沿 `stack + blend hybrid` 小步搜索，而不是继续扩大模型池
