# phase15_orig_fe_xgb_stack

- Status: `approved -> in_progress`
- Task: 新增 `original-data FE + bi/tri-gram TE + XGBoost` 低相关强单模，并接入新的二层 stacking 管线
- Created At: `2026-03-23 13:59:59 +08:00`
- Constraint: 训练与正式提交仅在 Kaggle 远程执行；本地仅做最小链路验证、配置维护、数据集整理与文档记录

## Current Anchor

1. 当前最佳 Public LB：`0.91608`
   - 提交：`phase10 stack oof v1`
2. 当前最佳本地 OOF：`0.9184734996`
   - 路线：`phase14 stronger stack pipeline v1/v2`
3. 当前阶段判断：
   - `phase11-14` 对现有模型池继续抠权重，已接近平台期
   - 下一轮应优先引入新的低相关底模，而不是继续微调现有二层权重

## Goal

1. 落地 `phase15_orig_fe_xgb` 单模 kernel：
   - `original-data transfer features`
   - `digit features`
   - `bi/tri-gram features`
   - `nested target encoding`
   - `single XGBoost`
2. 为单模提供本地 smoke 验证脚本，确保最小链路可跑通。
3. 落地 `phase15_stack_plus_orig_xgb` 二层 kernel，并复用 `phase14` 强特征 pipeline。
4. 准备新的 `phase15_stack_inputs` 数据集整理脚本，便于后续远程运行。

## Execution Plan

1. 创建 `kaggle_kernel/phase15_orig_fe_xgb/`
   - `train_phase15_orig_fe_xgb.py`
   - `config_phase15_orig_fe_xgb.json`
   - `config_phase15_orig_fe_xgb_smoke.json`
   - `kernel-metadata.json`
2. 创建 `scripts/smoke/smoke_phase15_orig_fe_xgb.py`
3. 创建 `scripts/prepare/prepare_phase15_stack_inputs.py`
4. 创建 `kaggle_dataset/phase15_stack_inputs/dataset-metadata.json`
5. 创建 `kaggle_kernel/phase15_stack_plus_orig_xgb/`
   - `stack_pipeline_search.py`
   - `stack_pipeline_config_v1.json`
   - `stack_pipeline_config_smoke.json`
   - `kernel-metadata.json`
6. 创建 `scripts/smoke/smoke_phase15_stack_plus_orig_xgb.py`
7. 本地执行最小 smoke，记录通过情况与阻塞点。

## Acceptance

1. `phase15_orig_fe_xgb` 本地 smoke 可运行并产出标准工件。
2. `phase15_stack_plus_orig_xgb` 本地 smoke 可运行并产出标准工件。
3. 代码结构可直接支撑后续 Kaggle 远程运行与正式提交。

## Progress Log

### 2026-03-23 13:59:59 +08:00

1. 新任务已创建并批准执行。
2. 已确认本轮主线为：
   - 先新增低相关底模 `phase15_orig_fe_xgb`
   - 再将其接入新的 stack 路线 `phase15_stack_plus_orig_xgb`
3. 代码实现进行中，待完成本地 smoke 后补充结果记录。

### 2026-03-23 14:23:37 +08:00

1. 已完成代码落地：
   - `kaggle_kernel/phase15_orig_fe_xgb/`
   - `kaggle_kernel/phase15_stack_plus_orig_xgb/`
   - `scripts/smoke/smoke_phase15_orig_fe_xgb.py`
   - `scripts/smoke/smoke_phase15_stack_plus_orig_xgb.py`
   - `scripts/prepare/prepare_phase15_stack_inputs.py`
2. 本地静态校验：
   - `python -m py_compile ...` 已通过
3. 单模 smoke：
   - 命令：`python scripts/smoke/smoke_phase15_orig_fe_xgb.py --max-train-rows 1200 --max-test-rows 500`
   - 结果：通过
   - 说明：本地无原始 Telco 数据时自动回退到 `train split` 作为参考分布，仅用于链路验证
4. stack smoke：
   - 命令：`python scripts/smoke/smoke_phase15_stack_plus_orig_xgb.py --max-train-rows 1200 --max-test-rows 500`
   - 结果：通过
   - Smoke best candidate：`phase15_smoke_full_pool | raw_only | logreg_newtoncg_c0p25`
   - Smoke OOF AUC：`0.916312226`
5. 输入整理脚本验证：
   - 命令：`python scripts/prepare/prepare_phase15_stack_inputs.py --phase15-output-dir .artifacts/smoke_phase15_orig_fe_xgb/output --dest-dir .artifacts/phase15_stack_inputs_preview`
   - 结果：通过
6. 当前状态：
   - 本地代码、配置、smoke、数据整理脚本均可用
   - 尚未进行 Kaggle 远程运行、kernel push 与正式提交

### 2026-03-23 14:37:05 +08:00

1. 已执行自动优化清理：
   - 删除 `phase15_orig_fe_xgb` 中未使用的 `LGBM/CatBoost` 遗留配置与训练实现
   - 收紧 `prepare_phase15_stack_inputs.py` 的输入白名单，只保留 `phase15 stack` 实际依赖文件
   - 同步 `phase15_stack_plus_orig_xgb` 默认远程配置与 `stack_pipeline_config_v1.json`
2. 优化后复验：
   - `python -m py_compile ...` 通过
   - `python scripts/smoke/smoke_phase15_orig_fe_xgb.py --max-train-rows 1200 --max-test-rows 500` 通过
   - `python scripts/prepare/prepare_phase15_stack_inputs.py --phase15-output-dir .artifacts/smoke_phase15_orig_fe_xgb/output --dest-dir .artifacts/phase15_stack_inputs_preview` 通过
   - `python scripts/smoke/smoke_phase15_stack_plus_orig_xgb.py --max-train-rows 1200 --max-test-rows 500` 通过
3. 优化结果判断：
   - 无功能回归
   - 代码职责更单一，后续远程运行时更不易出现配置漂移与无效分支

### 2026-03-23 15:02:51 +08:00

1. `phase15_orig_fe_xgb` 已完成 Kaggle 远程推送：
   - Kernel: `chicachan/ps-s6e3-phase15-orig-fe-xgb-v1`
   - Version: `1`
2. 远程状态结论：
   - `CANCEL_ACKNOWLEDGED`
   - Failure: `Your notebook was stopped because it exceeded the max allowed execution duration.`
3. 下载到的远程日志显示：
   - 原始 Telco 数据已正常挂载
   - 约 `196s / fold`
   - 已完成到 `Fold 3/10`
   - 说明问题是运行时长超限，而不是脚本逻辑或数据挂载错误
4. 已做的修正：
   - 保留 `config_phase15_orig_fe_xgb_v1_backup.json`
   - 将默认远程配置切换为 `v2 fastfit`
   - 主要改动：
     - `n_folds: 10 -> 5`
     - `inner_folds: 5 -> 3`
     - `top_cats_for_ngram`: `6 -> 5`
     - `n_estimators: 50000 -> 30000`
     - `learning_rate: 0.0063 -> 0.01`
     - `early_stopping_rounds: 500 -> 300`
5. 下一步：
   - 待主人确认后，推送 `phase15_orig_fe_xgb` v2 并再次远程运行

### 2026-03-23 15:42:49 +08:00

1. 已推送并完成 `phase15_orig_fe_xgb` 修正版远程运行：
   - Kernel: `chicachan/ps-s6e3-phase15-orig-fe-xgb-v1`
   - Version: `3`
2. 远程最终结果：
   - Status: `COMPLETE`
   - Config run name: `phase15_orig_fe_xgb_v2_fastfit`
   - OOF AUC: `0.9129815519`
   - Fold AUC:
     - `0.9130728917`
     - `0.9137780335`
     - `0.9130157833`
     - `0.9136994320`
     - `0.9113956466`
   - Fold mean: `0.9129923574`
   - Fold std: `0.0008571769`
   - Best iterations:
     - `1498`
     - `1505`
     - `1475`
     - `1470`
     - `1470`
3. 模型结论：
   - 该路线已能在 Kaggle 时间限制内完整跑完
   - 但 `OOF AUC` 仅 `0.91298`，显著低于当前项目可用强底模区间
   - 因此不建议继续推进到 `phase15 stack`
4. 失败判断：
   - 问题不再是运行时长，而是当前 `phase15` 特征实现版本本身不够强
   - 更合理的下一步不是把这个弱底模硬塞进 stack，而是重新设计更高质量的新底模路线
