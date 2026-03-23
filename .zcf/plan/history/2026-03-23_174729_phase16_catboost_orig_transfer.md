# phase16_catboost_orig_transfer

- Status: approved -> in_progress
- Task: 在 phase8_catboost_strong 骨架上引入更高保真的 original-data transfer 特征，验证是否形成可接入 phase14 的新强底模
- Created At: 2026-03-23 16:22:52 +0800
- Created At (ISO): 2026-03-23T16:22:52+0800
- Constraint: 训练与正式提交仅在 Kaggle 远程执行；本地仅做最小链路验证、配置维护、数据整理与文档记录

## Current Anchor

1. 当前最佳 Public LB：0.91608
   - 提交：phase10 stack oof v1
2. 当前最佳本地 OOF：0.9184734996
   - 路线：phase14 stronger stack pipeline
3. 当前最强单模：phase8 catboost strong v1
   - Public LB：0.91591
4. 上一轮失败路线：phase15_orig_fe_xgb
   - Kaggle 远程已完整跑通
   - OOF AUC：0.9129815519
   - 失败原因：不是运行时长，而是当前实现下模型强度不足

## Goal

1. 落地 phase16_catboost_orig_transfer 单模 kernel：
   - 复用 phase8 稳定 CatBoost 训练框架
   - 升级 original-data transfer 特征为更高保真版本
   - 保持 Kaggle 远程运行时长可控
2. 为单模提供本地 smoke 验证脚本，确保最小链路可跑通。
3. 为后续是否接入 phase14 stack 预留统一输出工件。
4. 仅在单模达到门槛时再继续推进 stack 输入整理与二层验证。

## Execution Plan

1. 归档 phase15 失败任务，并记录失败结论。
2. 创建 kaggle_kernel/phase16_catboost_orig_transfer/
   - train_phase16_catboost_orig_transfer.py
   - config_phase16_catboost_orig_transfer.json
   - config_phase16_catboost_orig_transfer_smoke.json
   - kernel-metadata.json
3. 创建 scripts/smoke/smoke_phase16_catboost_orig_transfer.py
4. 在训练脚本中实现：
   - base features + digit artifact features
   - high-fidelity original transfer features
   - leak-safe inner-fold target mean features
   - CatBoost GPU/CPU 双配置
5. 本地执行 py_compile 与 smoke 验证。
6. 完成本地验证后，再由主人确认是否进行 Kaggle 远程 push / run。

## Acceptance

1. phase16_catboost_orig_transfer 本地静态校验通过。
2. 若本机具备 catboost，则 smoke 可运行并产出标准工件。
3. 输出工件包含：
   - oof_phase16_catboost_orig_transfer.csv
   - submission_phase16_catboost_orig_transfer.csv
   - cv_metrics.json
   - run_summary.json
   - feature_importance.csv
4. 代码结构可直接支撑后续 Kaggle 远程运行。

## Progress Log

### 2026-03-23 16:22:52 +0800

1. 新任务已创建并批准执行。
2. 已确认本轮主线为：
   - 先做 phase16_catboost_orig_transfer 强单模验证
   - 单模达标后再决定是否接入 phase14 stack 管线
3. 当前实现进行中，待完成本地 smoke 后补充结果记录。

### 2026-03-23 16:24:43 +0800

1. 已完成 phase16 本地代码落地：
   - kaggle_kernel/phase16_catboost_orig_transfer/
   - scripts/smoke/smoke_phase16_catboost_orig_transfer.py
2. 本地静态校验：
   - python -m py_compile 已通过
3. 本地 smoke 结果：
   - 因本机未安装 catboost，训练型 smoke 自动跳过
   - 这是环境限制，不是脚本错误
4. 预处理链路补充验证：
   - 已在 1200/500 小样本上跑通 feature engineering 到训练前阶段
   - 结果：feature_count=156, num_cols=124, cat_cols=32
   - created_signals: freq=6, orig_single=6, orig_cross=6, orig_dist=21, orig_conditional=4, orig_quantile=18
5. 当前判断：
   - phase16 的本地脚手架、配置与输出约定已可用于 Kaggle 远程运行
   - 下一步若继续，需要主人确认执行 Kaggle kernel push / run

### 2026-03-23 17:26:31 +0800

1. 已完成 phase16 Kaggle 远程多轮验证。
2. 远程版本结果：
   - v1：超时
     - 原因：远程实际回退到脚本默认配置，单折训练过重
   - v2 fastfit：超时
   - v3 ultrafast：超时
   - v4 screen：超时
   - v5 default screen：完成
3. 关键根因定位：
   - Kaggle 远程未读取同目录 config 文件
   - 实际一直回退到脚本内 DEFAULT_CONFIG
   - 修复方式：将 DEFAULT_CONFIG 直接同步为压缩版配置，并在日志中打印 config 加载路径
4. v5 最终远程结果：
   - Kernel: chicachan/ps-s6e3-phase16-catboost-orig-transfer-v1
   - Version: 5
   - Run name: phase16_catboost_orig_transfer_v5_default_screen
   - Status: COMPLETE
   - OOF AUC: 0.9176360422
   - Fold AUC:
     - 0.9176750021
     - 0.9176044179
   - AUC std: 0.0000352921
   - Best iterations:
     - 1199
     - 1193
   - Feature count: 88
5. 当前结论：
   - 该路线在压缩到可运行预算后，OOF 仍低于 phase8 单模 0.9181653
   - 因此 phase16_catboost_orig_transfer 不应继续推进到 stack 接入
   - 但中途日志表明“更重的 original-transfer 版本”有潜在强度，只是在当前 Kaggle 时限下不可执行
6. 当前建议：
   - 结束 phase16 CatBoost 路线
   - 下一轮改为更轻的模型族复用这些高价值 transfer 信号，例如：LightGBM / XGBoost rebuild

### 2026-03-23 17:33:08 +0800

1. 已完成本轮代码级优化：
   - 将 add_frequency_features / add_orig_signal_features 改为批量收集后一次性 pd.concat
   - 新增 active_config.json 输出，记录实际生效配置与解析后的 config 路径
2. 优化后复验：
   - python -m py_compile 通过
   - 将 pandas PerformanceWarning 提升为 error 后，预处理链路仍可跑通
   - active_config.json 已验证可正常落盘
3. 优化结果判断：
   - 消除了远程日志中的主要 DataFrame fragmentation 风险
   - 后续 Kaggle 远程排查配置漂移会更直接
