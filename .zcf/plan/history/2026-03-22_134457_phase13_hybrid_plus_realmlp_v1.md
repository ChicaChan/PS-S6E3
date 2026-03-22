# Phase13 Hybrid Plus RealMLP V1

- Status: `approved -> in_progress`
- Task: 基于 `phase12 rank hybrid best` 继续叠加 `phase9_realmlp` 低权重增强，争取超过当前 Public LB best `0.91608`。
- Constraint:
  - 本地仅做最小 smoke test
  - 正式候选生成、提交与计分仅在 Kaggle 远程执行
  - 继续使用 `127.0.0.1:7890` 代理进行下载、上传与提交
  - 不扩模型池，只验证 `phase12 best + RealMLP` 小权重增强

## Objective

1. 实现 `phase13_hybrid_plus_realmlp` 搜索脚本与配置。
2. 重点验证 `rank` 小权重增强是否能把 OOF 从 `0.9184687240` 继续抬高。
3. 形成 1 个具备正式提交价值的候选并回收 Public LB。

## Execution Plan

1. 新建 `kaggle_kernel/phase13_hybrid_plus_realmlp`
   - `hybrid_plus_realmlp_search.py`
   - `phase13_config_v1.json`
   - `phase13_config_smoke.json`
   - `kernel-metadata.json`
2. 新建 `scripts/smoke/smoke_phase13_hybrid_plus_realmlp.py`
3. `hybrid_plus_realmlp_search.py` 支持：
   - 读取 `phase12 best` 与 `phase9_realmlp` 的 OOF / submission
   - 构建 `prob` / `rank` 小权重增强候选
   - 缓存 `rank` 结果，避免重复排序
   - 输出最佳 OOF / submission / 报告
4. 本地只执行 `py_compile + 抽样 smoke`
5. 准备 `phase13` 远程 dataset，随后在 Kaggle 远程执行并正式提交

## Notes

1. 当前本地证据：
   - `phase12 best` OOF AUC: `0.9184687240`
   - `phase12 best + phase9_realmlp` 最优局部测试：`0.9184704303`
   - 最强模式：`rank`
   - `RealMLP` 最佳低权重大约在 `0.02`
2. 本轮优先验证“低相关神经模型作为小权重增强”是否还能继续带来真实线上收益。
