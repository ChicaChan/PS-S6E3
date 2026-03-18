# PS-S6E3 Power BI EDA Dashboard

- Status: `approved -> completed`
- Task: 基于 `train.csv` 构建 Power BI EDA 可视化看板（仅交付 PBIX）
- Constraint:
  - 仅使用当前训练数据 `train.csv`
  - 不新增外部数据准备脚本
  - 看板目标偏 EDA

## Context Snapshot

1. 数据规模：`594,194` 行，`21` 列
2. 目标列：`Churn`（`No`: 77.4792%，`Yes`: 22.5208%）
3. 缺失值：`0`
4. 主要分析维度：
   - 类别：`gender`, `Partner`, `Dependents`, `InternetService`, `Contract`, `PaymentMethod` 等
   - 数值：`tenure`, `MonthlyCharges`, `TotalCharges`

## Approved Implementation Plan

1. 导入 `train.csv` 为单表模型（表名：`train`）
2. 完成字段类型校验（不做额外清洗）
3. 创建核心 DAX 度量
4. 创建 EDA 用分箱列（PBIX 内）
5. 构建 3 页看板：概览 / 类别分析 / 数值分析
6. 完成交互与口径校验并保存 `PS-S6E3_EDA.pbix`

## DAX Measures

```DAX
Total Rows = COUNTROWS(train)

Churn Yes = CALCULATE(COUNTROWS(train), train[Churn] = "Yes")

Churn No = CALCULATE(COUNTROWS(train), train[Churn] = "No")

Churn Rate = DIVIDE([Churn Yes], [Total Rows], 0)

Avg MonthlyCharges = AVERAGE(train[MonthlyCharges])

Avg TotalCharges = AVERAGE(train[TotalCharges])

Avg tenure = AVERAGE(train[tenure])
```

## Calculated Columns

```DAX
Tenure Band =
SWITCH(
    TRUE(),
    train[tenure] <= 12, "01-12",
    train[tenure] <= 24, "13-24",
    train[tenure] <= 36, "25-36",
    train[tenure] <= 48, "37-48",
    train[tenure] <= 60, "49-60",
    "61-72"
)

MonthlyCharges Band =
SWITCH(
    TRUE(),
    train[MonthlyCharges] <= 30, "<=30",
    train[MonthlyCharges] <= 50, "31-50",
    train[MonthlyCharges] <= 70, "51-70",
    train[MonthlyCharges] <= 90, "71-90",
    train[MonthlyCharges] <= 110, "91-110",
    ">110"
)

TotalCharges Band =
SWITCH(
    TRUE(),
    train[TotalCharges] <= 1000, "<=1k",
    train[TotalCharges] <= 2000, "1k-2k",
    train[TotalCharges] <= 3000, "2k-3k",
    train[TotalCharges] <= 4000, "3k-4k",
    train[TotalCharges] <= 6000, "4k-6k",
    ">6k"
)
```

## Page Design

1. Page 1 - `Overview`
   - KPI: `[Total Rows]`, `[Churn Yes]`, `[Churn Rate]`, `[Avg MonthlyCharges]`
   - 图表：`Churn` 占比 donut；`Contract` vs `Churn Rate` 柱图
   - 切片器：`gender`, `InternetService`, `PaymentMethod`

2. Page 2 - `Categorical EDA`
   - 图表：各类别字段对 `Churn Rate` 的条形图（降序）
   - 建议字段：`Contract`, `InternetService`, `PaymentMethod`, `OnlineSecurity`, `TechSupport`

3. Page 3 - `Numeric EDA`
   - 图表：`Tenure Band`, `MonthlyCharges Band`, `TotalCharges Band` 的样本量与 `Churn Rate`
   - 辅助图：`MonthlyCharges` 与 `TotalCharges` 按 `Churn` 的箱线图/散点

## Validation Checklist

1. `[Churn Yes] + [Churn No] = [Total Rows]`
2. 全样本 `Churn Rate` 应接近 `22.52%`
3. 各页切片器联动正常
4. 文件保存路径：`D:/workplace/kaggle/PS-S6E3/PS-S6E3_EDA.pbix`

## Progress Log

### 2026-03-18T18:01:52+0800

1. `PS-S6E3_EDA.pbix` 已完成并保存：`D:/workplace/kaggle/PS-S6E3/PS-S6E3_EDA.pbix`
2. 新增新手操作手册：`docs/powerbi_eda_beginner_guide.md`
3. 计划执行状态更新为 `completed`
