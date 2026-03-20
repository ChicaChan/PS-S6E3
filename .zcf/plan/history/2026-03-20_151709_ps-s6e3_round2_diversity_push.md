# PS-S6E3 Round-2 Diversity Push

- Status: `approved -> completed`
- Task: 新一轮冲分，目标是在当前最佳 `0.91591` 基础上继续提升
- Created At: `2026-03-19T11:15:10+0800`
- Constraint:
  - 训练与提交仅在 Kaggle 远程执行
  - 本地只做最小验证、OOF 分析、融合搜索和文档维护

## Current Best Snapshot

1. Best Public LB: `0.91606`
   - submission: `phase9 realmlp low-weight blend v1`
2. Strongest single model:
   - `phase8 catboost strong v1` -> `0.91591`
3. Current best blend:
   - `phase8_cat + phase6_cat + phase6_lgbm + phase3 + phase2 + low-weight RealMLP`
   - blend OOF AUC: `0.9184601`
4. Confirmed dead route:
   - `phase5_xgb_advanced`

## Round-2 Objective

1. 增强当前最强单模 `CatBoost`
2. 引入真正低相关的新模型族
3. 为二层 stacking 准备更有价值的 OOF 底座

## Hypotheses

1. 当前瓶颈不是树模型能力不足，而是模型同质化过高
2. `ORIG_proba_cross + pctrank_gap + conditional rank` 仍有机会提升 `CatBoost`
3. `RealMLP/TabM` 这类模型即使单模不超 `CatBoost`，也可能通过低相关性提升最终融合

## Execution Plan

### Track-A: CatBoost Strong-Feature v2

1. 基于现有 `phase6_diverse_tree` 增加更系统的原始数据迁移特征
2. 首批只引入高价值特征簇：
   - `ORIG_proba_cross`
   - `pctrank_gap`
   - `cond_pctrank`
3. 保持当前 fold/seed/评估逻辑不变
4. 远程训练后记录：
   - `cv_metrics_cat_v2.json`
   - `submission_cat_v2.csv`

### Track-B: RealMLP or TabM v1

1. 新建 Kaggle 远程训练目录
2. 在 Kaggle 环境中安装所需依赖
3. 复用现有特征工程与 fold 切分
4. 输出：
   - `oof_realmlp.csv` 或 `oof_tabm.csv`
   - `submission_realmlp.csv` 或 `submission_tabm.csv`

### Track-C: Round-2 Blend / Stack

1. 将 `phase6_cat_v1`、`phase6_cat_v2`、`phase6_lgbm_v1`、`phase2_fe_v1`、新模型 OOF 统一对齐
2. 先做相关性过滤，再尝试：
   - `prob blend`
   - `rank blend`
   - `Ridge/Logistic` 二层 stacking
3. 只提交 OOF 和 Public LB 同时合理的候选

## Success Criteria

1. 新单模至少满足以下之一：
   - OOF AUC 明显高于 `0.91806`
   - 或与 `phase6_cat_v1` 相关性明显更低
2. 新融合候选 Public LB 超过 `0.91591`

## Deliverables

1. Round-2 远程训练脚本
2. 新模型 OOF 与提交文件
3. 新一轮融合配置与结果报告
4. 更新后的项目级 memory 与任务记录

## Notes

1. 不再继续投入 `phase5`
2. 不再优先做现有高相关树模型的细碎权重微调
3. 所有新路线都必须优先证明“多样性价值”而不是“复杂度价值”

## Progress Log

### 2026-03-19T12:08:01+0800

1. Track-A code scaffold（已完成）
   - 新增 Kaggle 远程目录：
     - `kaggle_kernel/phase8_catboost_strong`
   - 新增文件：
     - `train_catboost_strong.py`
     - `config_catboost_strong.json`
     - `kernel-metadata.json`
     - `scripts/smoke/smoke_phase8_catboost_strong.py`
2. 这一版相对 `phase6_catboost v1` 的增强点：
   - `orig_single_mode` 支持 `all_categorical`
   - 新增 `pctrank_churn_gap_*`
   - 新增条件分位数特征：
     - `cond_pctrank_IS_TC`
     - `cond_pctrank_C_TC`
   - 保留 `ORIG_proba_cross` 与 leak-free TE 骨架
3. 本地验证：
   - `python -m py_compile kaggle_kernel/phase8_catboost_strong/train_catboost_strong.py` 通过
   - `python -m py_compile scripts/smoke/smoke_phase8_catboost_strong.py` 通过
   - smoke runner 成功执行
   - 当前本机未安装 `catboost`，因此运行时训练被安全跳过
4. 下一步：
   - 将 `phase8_catboost_strong` 推送到 Kaggle
   - 等待远程训练完成后回收 `cv_metrics.json` 与 `submission.csv`
   - 若单模优于 `0.91581`，再并入下一轮 blend / stack

### 2026-03-19T13:31:58+0800

1. Track-A 远程训练（已完成）
   - Kernel: `chicachan/ps-s6e3-catboost-strong-v1`
   - Status: `COMPLETE`
2. Phase-8 单模结果：
   - OOF AUC: `0.9181653`
   - 相比 `phase6 catboost v1` 的 `0.9180636`，提升 `0.00010`
   - 结论: 强特征版 CatBoost 有效，但提升幅度偏小
3. 与现有强模关系：
   - `phase8_cat` vs `phase6_cat` 相关性: `0.998622`
   - 结论: 这是同路线强化，不是全新多样性来源
4. Phase-8 融合复核（已完成）
   - 配置:
     - `kaggle_kernel/phase7_blend_oof/local_blend_config_phase8_candidates.json`
   - best OOF AUC: `0.9183841`
   - best method: `prob`
   - best weights:
     - `phase8_cat_v1`: `0.5238`
     - `phase6_cat_v1`: `0.1822`
     - `phase6_lgbm_v1`: `0.1674`
     - `phase3_ridge_xgb_v1`: `0.0960`
     - `phase2_fe_v1`: `0.0307`
   - 结论:
     - 相比当前最佳 blend OOF `0.9183020` 继续小幅提升
     - 具备实盘测试价值，但不能高估幅度
5. 当前推荐提交顺序：
   - `phase8_catboost_strong/output_v2/submission_cat.csv`
   - `phase7_blend_oof/output_phase8_candidates/submission_blend_opt.csv`
6. Phase-8 单模正式提交（已完成）
   - Timestamp: `2026-03-19T13:34:53+0800`
   - 提交文件:
     - `kaggle_kernel/phase8_catboost_strong/output_v2/submission_cat.csv`
   - Kaggle message:
     - `phase8 catboost strong v1`
   - Public LB:
     - `0.91591`
   - 结果判断:
     - 与当前最佳 `0.91591` 持平
     - 说明 `phase8` 的特征增强路线是有效的，但暂未单独突破现有最佳融合
7. 下一优先级:
   - `phase7_blend_oof/output_phase8_candidates/submission_blend_opt.csv`
8. Phase-8 融合正式提交（已完成）
   - Timestamp: `2026-03-19T13:36:54+0800`
   - 提交文件:
     - `kaggle_kernel/phase7_blend_oof/output_phase8_candidates/submission_blend_opt.csv`
   - Kaggle message:
     - `phase7 phase8 candidate blend opt v1`
   - Public LB:
     - `0.91602`
   - 结果判断:
     - 成功超过此前最佳 `0.91591`
     - 证明 `phase8_cat` 虽然与 `phase6_cat` 高相关，但作为强化版纳入受限融合后仍能产生稳定正增益
9. 当前阶段结论:
   - Round-2 Track-A 已被验证有效
   - 下一轮若要继续上分，重点应切到真正低相关的新模型族，而不是继续在 CatBoost 同族内微调

### 2026-03-19T14:03:06+0800

1. Track-B code scaffold（已完成）
   - 新增 Kaggle 远程目录：
     - `kaggle_kernel/phase9_realmlp_tabm_diverse`
   - 新增文件：
     - `train_realmlp_tabm_diverse.py`
     - `config_realmlp_tabm_diverse.json`
     - `kernel-metadata.json`
     - `scripts/smoke/smoke_phase9_realmlp_tabm_diverse.py`
2. 这一版相对 `phase8` 的核心变化：
   - 复用 `phase8` 已验证的强特征链路：
     - `add_base_features`
     - `add_ngram_features`
     - `add_orig_signal_features`
     - leak-free target mean encoding
   - 删除树模型训练主干，仅保留真正低相关的新模型族：
     - `RealMLP_TD_Classifier`
     - `TabM_D_Classifier`
   - 新增 `ensure_pytabkit`，允许 Kaggle 侧按需安装 `pytabkit`
   - `kernel-metadata.json` 已开启：
     - `enable_gpu: true`
     - `enable_internet: true`
3. 本地验证：
   - `python -m py_compile kaggle_kernel/phase9_realmlp_tabm_diverse/train_realmlp_tabm_diverse.py` 通过
   - `python -m py_compile scripts/smoke/smoke_phase9_realmlp_tabm_diverse.py` 通过
   - smoke runner 成功执行
   - 当前本机未安装 `pytabkit`，因此运行时训练被安全跳过
4. 下一步：
   - 将 `phase9_realmlp_tabm_diverse` 推送到 Kaggle
   - 等待远程训练完成后回收：
     - `oof_realmlp.csv`
     - `oof_tabmd.csv`
     - `oof_ensemble.csv`
     - `phase9_report.json`
   - 训练完成后先做：
     - 与 `phase6_cat_v1` / `phase8_cat_v1` 的 OOF 相关性分析
     - 再决定是否进入 Track-C 融合和正式提交

### 2026-03-19T14:54:41+0800

1. Track-B 远程推送（已完成）
   - Kernel:
     - `chicachan/ps-s6e3-realmlp-tabm-diverse-v1`
   - Push 结果:
     - `versionNumber = 1`
     - `invalidDatasetSources = []`
     - `invalidCompetitionSources = []`
   - URL:
     - `https://www.kaggle.com/code/chicachan/ps-s6e3-realmlp-tabm-diverse-v1`
2. 远程运行状态（进行中）
   - 多轮轮询结果持续为：
     - `KernelWorkerStatus.RUNNING`
   - 当前无 failure message
   - 当前 `kernels_output` 尚无可下载文件
3. 当前判断：
   - 远程任务未秒级失败，环境与数据挂载大概率正常
   - 但 `RealMLP + TabM + 5 folds + 默认参数` 的总训练时长明显偏长
   - 若后续超过可接受时间或出现超时，优先准备轻量版 v2：
     - 限制 `n_epochs`
     - 显式设置 `batch_size`
     - 先单跑 `RealMLP` 再决定是否追加 `TabM`

### 2026-03-19T16:24:04+0800

1. `v1` 取消尝试（已完成，未成功）
   - 已核查本机可用的 Kaggle 官方 CLI 与 Python SDK：
     - CLI 仅暴露 `list / files / init / push / pull / output / status`
     - SDK 仅暴露 `save / status / output / pull / list`
   - 未发现公开可用的 `cancel / stop / interrupt` kernel session 接口
   - 已尝试高概率内部 endpoint 探测：
     - `/api/v1/kernels/cancel`
     - `/api/v1/kernels/stop`
     - `/api/v1/kernels/interrupt`
     - 以及对应 `session/*` 形式
   - 结果：
     - 全部 `404`
   - 当前结论：
     - 浮浮酱无法通过当前本地官方接口直接取消 `v1`
2. Track-B 轻量版 `v2`（已完成）
   - 保留 `phase9_realmlp_tabm_diverse` 路线与强特征骨架
   - 收敛策略：
     - `n_folds: 3`
     - `inner_folds: 3`
     - `enable_tabm: false`
     - 先只验证 `RealMLP`
   - `RealMLP` 显式限缩参数：
     - `n_epochs: 64`
     - `batch_size: 2048`
     - `hidden_sizes: [128, 128, 128]`
     - `lr: 0.04`
     - `use_ls: false`
     - `verbosity: 1`
   - 新增可观测性：
     - 主流程改为 line-buffered stdout
     - fold 开始/结束即时日志
     - 每 fold 写 `heartbeat_realmlp.json`
3. 本地验证：
   - `python -m py_compile kaggle_kernel/phase9_realmlp_tabm_diverse/train_realmlp_tabm_diverse.py` 通过
   - `python -m py_compile scripts/smoke/smoke_phase9_realmlp_tabm_diverse.py` 通过
   - smoke runner 成功执行
   - 当前本机未安装 `pytabkit`，运行时训练仍安全跳过
4. 下一步：
   - 若主人确认，直接 push 同一 kernel 生成 `version 2`
   - 重点观察：
     - 是否能在合理时间内完成
     - `heartbeat_realmlp.json` 与折间日志是否可见

### 2026-03-19T16:59:55+0800

1. Track-B `v2` 远程训练（已完成）
   - Kernel:
     - `chicachan/ps-s6e3-realmlp-tabm-diverse-v1`
   - Version:
     - `2`
   - Status:
     - `COMPLETE`
2. `v2` 输出文件（已回收）
   - `cv_metrics_realmlp.json`
   - `heartbeat_realmlp.json`
   - `oof_realmlp.csv`
   - `oof_ensemble.csv`
   - `submission_realmlp.csv`
   - `submission.csv`
   - `phase9_report.json`
3. `v2` 结果判断：
   - `RealMLP` OOF AUC:
     - `0.9147087`
   - 对比现有强树模：
     - `phase6_cat`: `0.9180636`
     - `phase8_cat`: `0.9181653`
   - 相关性：
     - `RealMLP vs phase6_cat`: `0.980665`
     - `RealMLP vs phase8_cat`: `0.980994`
   - 结论：
     - `v2` 已证明“可跑通、可观测、时长可控”
     - 但当前单模强度明显不足，且相关性虽然低于树模同族，但还不够低
     - 暂不建议将 `submission_realmlp.csv` 直接作为正式提交候选
4. 下一步推荐：
   - 不做单模提交
   - 若继续利用该路线，只建议做一次受限融合试验：
     - 给 `RealMLP` 很低权重
     - 与 `phase6_cat / phase8_cat / phase6_lgbm / phase3 / phase2` 一起做 OOF 约束搜索

### 2026-03-19T18:06:59+0800

1. `RealMLP` 低权重融合试验（已完成）
   - 基线:
     - `phase7_blend_oof/output_phase8_candidates/blend_report.json`
     - best OOF AUC: `0.9183840643`
   - 试验方式:
     - 固定当前最佳五模权重
     - 对 `phase9_realmlp_v2` 做一维概率混合网格搜索
     - 搜索区间: `[0.0, 0.2]`
     - 搜索步数: `201`
2. 试验结果：
   - 最佳 `RealMLP` 权重:
     - `0.124`
   - 新 blend OOF AUC:
     - `0.9184601341`
   - 相对基线提升:
     - `+0.0000761`
3. 最终权重：
   - `phase8_cat_v1`: `0.4588309006`
   - `phase6_cat_v1`: `0.1595808377`
   - `phase2_fe_v1`: `0.0269030022`
   - `phase3_ridge_xgb_v1`: `0.0840683018`
   - `phase6_lgbm_v1`: `0.1466169578`
   - `phase9_realmlp_v2`: `0.1240000000`
4. 输出文件：
   - `kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/submission_blend_opt.csv`
   - `kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/oof_blend_opt.csv`
   - `kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/blend_report.json`
   - `kaggle_kernel/phase7_blend_oof/local_blend_grid_phase9_realmlp_summary.json`
5. 当前判断：
   - `RealMLP` 虽然单模偏弱，但作为低权重辅助源确实带来正向 OOF 增益
   - 增益幅度较小，但已经超过此前 best blend OOF
   - 这版候选具备正式提交测试价值
