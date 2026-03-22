# phase14_stronger_stack_pipeline_v1

## 背景

- 当前最佳 Public LB：`0.91608`
- 当前最佳本地 OOF：`0.9184704345`，来自 `phase13_hybrid_plus_realmlp_v1`
- 既有 `phase11 / phase12 / phase13` 的窄范围权重微调已接近平台期
- 下一优先方向改为更系统的二层 stacking feature pipeline，而不是继续抠单一 blend 权重

## 目标

- 主目标：产出 `phase14` stronger stack pipeline，并在 Kaggle 云端完成正式运行与提交
- 成功标准：
  - Public LB 严格高于 `0.91608`
  - 本地 OOF 严格高于 `0.9184704345`

## 输入基座

1. `phase13_hybrid_best_v1`
2. `phase10_stack_best_v1`
3. `phase7_blend_best_v1`
4. `phase8_cat_v1`
5. `phase9_realmlp_v2`

## 执行步骤

1. 创建 `phase14` kernel 主脚本，支持：
   - 读取本地 / Kaggle 配置
   - 自动检测 Kaggle dataset 输入根目录
   - 读取并对齐多个 OOF / submission
   - 预计算 `raw / rank / logit`
   - 构造统计特征、anchor gap 特征、pairwise absdiff 特征
   - 运行多组 meta model 的 OOF stacking 搜索
   - 输出最佳 `oof`、`submission`、候选汇总与报告
2. 创建本地配置、smoke 配置、kernel metadata、dataset metadata
3. 创建最小 smoke 脚本，只抽样少量训练 / 测试行做功能验证
4. 本地仅执行最小验证：
   - `py_compile`
   - smoke 抽样运行
5. 将 `phase14` 输入文件复制到 `kaggle_dataset/phase14_stack_pipeline_inputs/`
6. 通过 `127.0.0.1:7890` 代理上传 Kaggle dataset，并推送 kernel
7. 轮询远程状态，下载输出并检查：
   - `stack_pipeline_report.json`
   - `candidate_summary.json`
   - `oof_stack_pipeline_best.csv`
   - `submission_stack_pipeline_best.csv`
8. 若远程结果有效，则正式提交并回收 Public LB 分数

## 本地验证约束

- 本地只做最小限度测试
- 不在本地执行完整训练
- 正式训练、融合搜索与提交全部放到 Kaggle 云端执行

## 预期产物

- `kaggle_kernel/phase14_stronger_stack_pipeline/stack_pipeline_search.py`
- `kaggle_kernel/phase14_stronger_stack_pipeline/stack_pipeline_config_v1.json`
- `kaggle_kernel/phase14_stronger_stack_pipeline/stack_pipeline_config_smoke.json`
- `kaggle_kernel/phase14_stronger_stack_pipeline/kernel-metadata.json`
- `kaggle_dataset/phase14_stack_pipeline_inputs/dataset-metadata.json`
- `scripts/smoke/smoke_phase14_stronger_stack_pipeline.py`

## 执行结果

- `phase14 v1` 已完成远程运行与正式提交
- `phase14 v1` 最佳候选：
  - `stack_plus_diversity + full_linear + logreg_newtoncg_c0p25`
  - `OOF AUC = 0.9184734996`
  - `Public LB = 0.91607`
- `phase14 v2` 已完成日志增强与窄搜索优化：
  - 新增 `progress.log`
  - 新增 `candidate_progress.jsonl`
  - 新增 `best_candidate_snapshot.json`
- `phase14 v2` 远程结果与 `phase14 v1` 最优候选一致：
  - 最优仍为 `stack_plus_diversity + full_linear + logreg_newtoncg_c0p25`
  - `OOF AUC = 0.9184734996`
  - 未产生比 `phase14 v1` 更优的本地 OOF

## 当前结论

- 本轮完成了系统 stacking 管线与日志能力补强
- 但未达到主目标 `Public LB > 0.91608`
- 现阶段最合理的下一步不是继续调 `phase14` 线性二层，而是转向新的增益来源，例如：
  - 引入新的强多样性底模
  - 做更高质量的 OOF stacking 输入
  - 改为更强的 feature pipeline / target encoding / tabular NN 多样性方向
