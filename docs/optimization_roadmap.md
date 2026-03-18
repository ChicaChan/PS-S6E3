# PS-S6E3 优化路线图

## P1: 稳健增强（低风险，高性价比）

1. 调参网格：
   - `learning_rate / max_depth / min_child_weight / colsample_bytree / subsample`
2. TE 扩展：
   - 增加 `std/min/max` 聚合统计，保留 OOF 防泄漏流程
3. 伪标签门控：
   - 阈值从 `0.995` 扫描到 `0.999`
   - 每 fold 仅在验证 AUC 提升时启用

预期收益：提升基础单模表现，保持实现复杂度可控。

## P2: 两阶段建模（中风险，中高收益）

1. Stage-1：
   - `Ridge` 使用 OHE + 数值特征
2. Stage-2：
   - 将 Ridge 预测作为额外特征输入 XGB
3. 验证策略：
   - 同一 outer fold 内完成两阶段，避免信息泄漏

预期收益：增强线性+非线性互补，通常可优于单一 XGB。

## P3: 融合策略（中风险，高收益）

1. 多份 submission 统一对齐 `id`
2. 尝试：
   - `mean`
   - `weighted_rank`
3. 权重来源：
   - 优先 OOF 表现和模型差异性

预期收益：后期冲榜常见提升手段，但需控制 Public LB 过拟合。

## 试验记录建议

1. 每次实验记录：
   - 配置文件快照
   - OOF AUC 与 fold 方差
   - 提交文件名与 LB 分数
2. 固定随机种子后再比较方案，避免噪声误判。
