# PS-S6E3: 基于原始数据迁移特征与受限融合的客户流失预测方案

## 摘要

本文总结 `Kaggle Playground Series - Season 6 Episode 3` 当前项目方案。赛题目标是基于合成化的电信客户数据预测 `Churn`，评价指标为 `ROC AUC`。本项目遵循“远程训练、局部验证、本地受限融合”的工程约束，逐步构建了 `XGBoost + leak-free target encoding` 基线、特征增强分支、两阶段 `Ridge -> XGB` 分支、`CatBoost/LGBM` 多样性分支，以及基于 OOF 相关性过滤的受限融合分支。当前最佳 Public LB 为 `0.91606`，来自在 `phase7 phase8 candidate blend opt v1` 基础上低权重注入 `RealMLP` 的六模概率融合。实验表明，真正有效的冲分因素并不是继续微调同构 XGBoost，而是引入更强的原始数据迁移信号和更低相关的新模型族，并以受限权重方式并入现有强融合。

## 1. 研究背景

### 1.1 任务定义

- 比赛：`playground-series-s6e3`
- 任务类型：二分类
- 目标列：`Churn`
- 评价指标：`ROC AUC`

### 1.2 数据背景

- 训练集：`594,194` 行，`21` 列
- 测试集：`254,655` 行，`20` 列
- 数据形态：主赛题数据为合成数据，但与经典 Telco Customer Churn 分布高度相关
- 外部参考：原始 Telco 数据可作为分布参考源，用于构建 `ORIG_proba`、条件分位数和分布偏移类特征

### 1.3 工程约束

- 所有模型训练与正式提交均在 Kaggle 远程执行
- 本地仅用于脚本冒烟、OOF 分析、融合搜索和提交格式校验
- 冲榜策略必须以可复现为前提，避免仅依赖 Public LB 波动

## 2. 方法总览

本项目当前方案按 5 个阶段演化：

1. 基线阶段：`XGBoost + leak-free target encoding`
2. 稳健增强阶段：伪标签门控与轻量特征增强
3. 两阶段阶段：`Ridge -> XGB`
4. 多样性阶段：`CatBoost + LGBM`
5. 融合阶段：OOF 约束下的相关性过滤与权重搜索

核心原则如下：

- 所有目标编码都必须在外层 fold 内进行内层 OOF 编码，避免泄漏
- 伪标签只在 fold 级别 AUC 提升时启用
- 融合前先做相关性过滤，避免高 AUC 但同构的模型互相“重复加权”

## 3. 模型构建

### 3.1 Baseline: XGBoost + 防泄漏目标编码

基线模型使用：

- 树模型：`XGBoost`
- 编码方式：对高价值类别变量做 leak-free target encoding
- 验证方式：外层 5-fold，内层 OOF 编码

该基线的优点是实现稳定、训练成本低、便于后续所有分支复用。它构成了后续 `phase1/phase2/phase3` 的公共底座。

### 3.2 Phase-1: 伪标签门控

尝试对高置信度测试样本进行伪标签扩充，扫描阈值 `0.995/0.997/0.999`。结论是：

- 低阈值下大多数 fold 没有带来稳定增益
- 高阈值 `0.999` 仅带来非常有限收益
- 单独依赖伪标签并不能构成明显突破点

因此，伪标签在本项目中仅保留为辅助机制，而非主增益来源。

### 3.3 Phase-2: 特征增强

这一阶段引入更接近高分公开方案的轻量增强：

- `ORIG_proba` 映射特征
- 有信息密度的类别交叉
- 受限的统计映射扩展

其目标不是大幅改变模型族，而是在不破坏基线稳定性的前提下，注入来自原始 Telco 数据的先验。

### 3.4 Phase-3: 两阶段 Ridge -> XGB

该路线先用线性模型抽取一层可泛化信号，再将一级预测作为二级树模型输入：

- Stage-1：`Ridge`，使用 OHE + 数值特征
- Stage-2：`XGBoost`，额外接收 `ridge_pred`

它提供了一定结构多样性，并在早期取得了优于普通单模 XGB 的线上表现。

### 3.5 Phase-6: Diverse Tree 分支

这是当前最有效的新模型分支，包含：

- `CatBoost`
- `LightGBM`
- `CatBoost + LightGBM` 简单集成

关键结果：

- `CatBoost` OOF AUC：`0.9180636`
- `LightGBM` OOF AUC：`0.9158487`
- Phase-6 Ensemble OOF AUC：`0.9179283`

其中 `CatBoost` 是当前最强单模型，并成功将 Public LB 从 `0.91407` 提升到 `0.91581`。

### 3.6 Phase-7: OOF 约束融合

融合不直接基于 Public LB，而先依据 OOF 做两步处理：

1. 按单模 AUC 排序
2. 用相关性阈值过滤过高相关的候选
3. 在剩余模型上搜索 `rank/prob` 权重

当前最优融合配置（截至 `2026-03-20`）：

- 基线五模：`phase8_cat`, `phase6_cat`, `phase6_lgbm`, `phase2_fe`, `phase3`
- 在基线五模最优概率权重上，额外注入 `phase9_realmlp_v2`
- `RealMLP` 最优注入权重：`0.124`
- 最优权重：
  - `phase8_cat`: `0.4588`
  - `phase6_cat`: `0.1596`
  - `phase2_fe`: `0.0269`
  - `phase3_ridge_xgb`: `0.0841`
  - `phase6_lgbm`: `0.1466`
  - `phase9_realmlp_v2`: `0.1240`

该方案取得：

- OOF AUC：`0.9184601`
- Public LB：`0.91606`

## 4. 实验结果

### 4.1 关键阶段结果

| 阶段 | 代表方案 | OOF AUC | Public LB | 结论 |
| --- | --- | ---: | ---: | --- |
| Baseline | remote baseline xgb te v1 | `0.9162936` | `0.91384` | 稳定起点 |
| Phase-1 | pseudo label th999 | `0.9162921` | `0.91386` | 边际收益很小 |
| Phase-2 | fe v1 pseudo999 | `0.9163282` | `0.91387` | 轻微提升 |
| Phase-3 | ridge xgb v1 | `0.9162443` | `0.91400` | 线上有效 |
| Phase-4 | blend eq/opt rank | - | `0.91407` | 首次有效融合 |
| Phase-5 | xgb advanced v1/v2 | `0.9258733`/`0.9098060` | `0.89306` | 失败路线 |
| Phase-6 | catboost v1 | `0.9180636` | `0.91581` | 当前最强单模 |
| Phase-6 | ensemble v1 | `0.9179283` | `0.91567` | 不如单模 CatBoost |
| Phase-7 | phase6 candidate blend opt v1 | `0.9183020` | `0.91591` | 第一版有效最优融合 |
| Phase-8 | phase8 candidate blend opt v1 | `0.9183841` | `0.91602` | CatBoost 强化版并入后继续提升 |
| Phase-9 | realmlp low-weight blend v1 | `0.9184601` | `0.91606` | 当前最佳 |

### 4.2 失败案例分析

`phase5_xgb_advanced` 是一个重要反例。

- v1 因未正确挂载原始 Telco 数据，导致参考分布错误
- v2 修复挂载后，OOF 仅 `0.909806`
- 说明这一实现路线本身并不成立，而非单纯数据源配置问题

该结论的重要性在于：高 OOF 并不必然对应可泛化 Public LB，特别是在参考分布构造和目标编码链路较复杂时。

## 5. 从高分公开方案得到的启示

基于本地研究目录中的高分代码，当前可以稳定提炼出 4 个共性：

### 5.1 原始数据迁移信号是主增益源

高分单模大量使用：

- `ORIG_proba_*`
- `ORIG_proba_cross`
- `pctrank_*`
- `pctrank_gap_*`
- 条件分位数特征

这类特征的作用不是简单“补充列”，而是把原始 Telco 数据的标签条件分布迁移到合成赛题空间。

### 5.2 leak-free 编码比表面上的复杂特征更重要

公开高分方案几乎都显式使用了内层 OOF 的目标编码或等价的防泄漏逻辑。这个部分如果做错，即使特征很多，也会在 Public LB 上迅速塌陷。

### 5.3 真正的突破来自模型多样性

从高分代码看，后期稳定出现的模型族包括：

- `CatBoost`
- `LightGBM`
- `RealMLP`
- `TabM`
- 更激进的神经表格模型或图模型

对我们当前项目而言，`CatBoost` 已经证明有效，下一步的真正空白是 `RealMLP/TabM` 一类与树模型低相关的新模型。

### 5.4 后期融合要靠“低相关强模型”，不是盲目加模型数

我们自己的实验已经验证：

- `phase3` 与 `phase2` 对现有最强模型相关性过高
- 直接全塞进融合，收益几乎没有
- 先做相关性过滤，再做小规模搜索，才有机会稳定转化为 LB 提升

## 6. 当前最优比赛方案

截至 `2026-03-20`，当前推荐的比赛主线为：

1. 单模基石：`phase8 CatBoost strong`
2. 辅助差异模型：`phase6 CatBoost`, `phase6 LGBM`, `phase2 FE`, `phase3`
3. 新多样性补丁：`phase9 RealMLP v2`
4. 最终提交：基于现有最佳五模概率融合，低权重注入 `RealMLP`

该方案的本质不是“大型堆模”，而是：

- 用 `phase8/phase6 CatBoost` 吃下当前最强树模型收益
- 用 `LGBM + Phase2 + Phase3` 提供稳健辅助信号
- 用 `RealMLP` 作为低权重多样性补丁提供最终小幅增益
- 用 OOF 约束避免线上过拟合

## 7. 下一轮冲分方案

下一轮目标不应再围绕当前 blend 权重微调，而应围绕“新增多样性”和“增强原始迁移信号”展开。

### 7.1 优先级 A：强特征版 CatBoost v2

方向：

- 将高分方案中的 `ORIG_proba_cross`、`pctrank_gap`、条件分位数特征系统化引入
- 控制新增特征数，优先做高信息密度子集
- 保持当前验证框架不变

原因：

- 这是对当前最强单模的直接增强
- 工程风险低于引入全新深度模型
- 一旦有效，可直接与现有 blend 兼容

### 7.2 优先级 B：RealMLP / TabM 新模型族

方向：

- 使用 Kaggle 远程环境安装或引入对应依赖
- 复用当前特征框架和 fold 切分
- 目标不是单模一定超过 CatBoost，而是获得低相关高质量 OOF

原因：

- 当前项目最大短板不是缺一个更像 XGB 的模型，而是缺一个真正低相关的强模型
- 高分公开方案已证明这类模型在该赛题中有效

### 7.3 优先级 C：二层 stacking

方向：

- 使用 `phase6_cat`, `phase6_lgbm`, `phase2_fe` 以及后续新模型 OOF
- 构建轻量 meta learner，例如 `Ridge/Logistic/XGB small`
- 严格使用 OOF 到 OOF 的二层训练，避免泄漏

原因：

- 当前简单线性融合已经逼近极限
- 若有新的低相关模型进入，stacking 比直接 rank/prob 平均更可能吃到结构增益

### 7.4 提交顺序建议

下一轮建议按以下顺序做实验：

1. `CatBoost strong-feature v2`
2. `RealMLP/TabM v1`
3. `CatBoost v2 + RealMLP/TabM` 受限融合
4. 二层 stacking

## 8. 结论

本项目当前已经完成从基线到有效冲分路线的验证，且 Public LB 已达到 `0.91606`。现阶段最重要的判断是：

- `phase5` 已证伪，不再投入
- `phase8 CatBoost` 是现阶段最优单模主线
- `RealMLP` 不适合作为独立提交，但作为低权重辅助源是有效的
- 小幅有效提升来自“强树模型主干 + 低相关受限融合”
- 下一轮要继续上分，必须继续引入新的模型多样性，而不是继续在高相关树模型之间内耗

换言之，下一阶段的重点不是再做一个“更复杂的 XGB”，而是继续构造能与 `phase8/phase6 CatBoost` 真正互补的新模型，并用更严格的权重约束将其并入现有最优融合喵～
