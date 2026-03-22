# Phase11 Stack Blend Hybrid V1

- Status: `approved -> in_progress`
- Task: 基于 `phase10 stack best` 与当前最优 `reference blend` 做二元 hybrid 小步搜索，争取继续提升 Public LB。
- Constraint:
  - 本地仅做最小 smoke test
  - 正式候选生成、提交与计分仅在 Kaggle 远程执行
  - 下载、上传与提交继续通过 `127.0.0.1:7890` 代理

## Objective

1. 实现 `phase11_stack_blend_hybrid` 脚本与配置，支持 `prob` / `rank` 二元融合搜索。
2. 以当前最优线上结果为基线：
   - Public LB: `0.91608`
   - reference blend OOF AUC: `0.9184677453`
3. 产出 1 个具备正式提交价值的 hybrid 候选。

## Execution Plan

1. 新建 `kaggle_kernel/phase11_stack_blend_hybrid`
   - `hybrid_search.py`
   - `hybrid_config_v1.json`
   - `hybrid_config_smoke.json`
   - `kernel-metadata.json`
2. 新建 `scripts/smoke/smoke_phase11_stack_blend_hybrid.py`
3. `hybrid_search.py` 支持：
   - 读取 `reference blend` 与 `phase10 stack best` 的 OOF / submission
   - 构建 `prob` / `rank` 二元融合候选
   - 扫描 `stack_weight` 的小步权重空间
   - 输出最佳候选 OOF / submission / 报告
4. 本地只执行最小 smoke：
   - 抽样少量 train/test 行
   - 跑通脚本链路
   - 校验输出格式
5. 准备 `phase11` 远程输入 dataset，随后在 Kaggle 远程执行并正式提交。

## Notes

1. 当前最强证据：
   - `phase10 stack best` OOF AUC: `0.9184538058`
   - `reference blend` OOF AUC: `0.9184677453`
   - 本地二元线性融合最佳点约为 `stack_weight=0.18`
   - 对应 OOF AUC: `0.9184685096`
2. 本轮优先验证 `stack + blend hybrid` 的真实线上收益，不扩模型池。
