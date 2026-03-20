# PS-S6E3 Power BI 新手操作手册（EDA）

目标：用 `train.csv` 做出 3 页 EDA 看板，并保存为 `PS-S6E3_EDA.pbix`。

参考归档计划文件：
- [简单EDA](D:/workplace/kaggle/PS-S6E3/.zcf/plan/history/2026-03-19_103723_简单EDA.md)
- [EDA优化](D:/workplace/kaggle/PS-S6E3/.zcf/plan/history/2026-03-20_170559_EDA优化.md)

版本管理建议：
- Git 仓库优先跟踪 `PS-S6E3_EDA_Optimized.pbip`、`PS-S6E3_EDA.Report/`、`PS-S6E3_EDA.SemanticModel/`
- `PBIX` 更适合作为最终打开与分发文件，不作为主要源码管理载体

## 1. 导入数据

1. 打开 `Power BI Desktop`。
2. 点击 `获取数据` -> `文本/CSV`。
3. 选择文件：`D:/workplace/kaggle/PS-S6E3/train.csv`。
4. 在预览窗口点 `加载`（不是“转换数据”）。
5. 右侧字段窗格应出现表 `train`。

## 2. 检查字段类型

1. 左侧点击 `数据` 视图（表格图标）。
2. 点 `train` 表，逐列检查：
   - 整数：`id`、`SeniorCitizen`、`tenure`
   - 小数：`MonthlyCharges`、`TotalCharges`
   - 文本：其余列（含 `Churn`）
3. 如果类型不对：选中列 -> 上方 `列工具` -> `数据类型` 修改。

## 3. 创建度量值（Measures）

在右侧字段窗格里，右键 `train` -> `新建度量值`，逐条粘贴以下 DAX：

```DAX
Total Rows = COUNTROWS(train)
```

```DAX
Churn Yes = CALCULATE(COUNTROWS(train), train[Churn] = "Yes")
```

```DAX
Churn No = CALCULATE(COUNTROWS(train), train[Churn] = "No")
```

```DAX
Churn Rate = DIVIDE([Churn Yes], [Total Rows], 0)
```

```DAX
Avg MonthlyCharges = AVERAGE(train[MonthlyCharges])
```

```DAX
Avg TotalCharges = AVERAGE(train[TotalCharges])
```

```DAX
Avg tenure = AVERAGE(train[tenure])
```

格式设置：
1. 选中 `Churn Rate` -> `度量值工具` -> `格式` 设为 `百分比`，保留 2 位小数。
2. `Avg MonthlyCharges`、`Avg TotalCharges` 设为小数 2 位。

## 4. 创建计算列（分箱）

右键 `train` -> `新建列`，逐条粘贴：

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
```

```DAX
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
```

```DAX
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

## 5. 搭建 3 个页面

### Page 1: Overview

1. 新建页面并重命名为 `Overview`。
2. 添加 4 个 `卡片`：
   - `[Total Rows]`
   - `[Churn Yes]`
   - `[Churn Rate]`
   - `[Avg MonthlyCharges]`
3. 添加 `圆环图`：
   - 图例：`Churn`
   - 值：`Total Rows`
4. 添加 `簇状条形图`（合同类型流失率）：
   - Y 轴：`Contract`
   - X 轴：`Churn Rate`
5. 添加 3 个切片器：
   - `gender`
   - `InternetService`
   - `PaymentMethod`

### Page 2: Categorical EDA

1. 新建页面命名 `Categorical EDA`。
2. 做 5 个条形图（都使用 `Churn Rate`）：
   - `Contract` vs `Churn Rate`
   - `InternetService` vs `Churn Rate`
   - `PaymentMethod` vs `Churn Rate`
   - `OnlineSecurity` vs `Churn Rate`
   - `TechSupport` vs `Churn Rate`
3. 每个图点右上角 `...` -> 排序按 `Churn Rate` 降序。

### Page 3: Numeric EDA

1. 新建页面命名 `Numeric EDA`。
2. 添加 3 个簇状柱形图：
   - `Tenure Band` + `Total Rows`
   - `MonthlyCharges Band` + `Total Rows`
   - `TotalCharges Band` + `Total Rows`
3. 再复制这 3 个图各一份，把值改成 `Churn Rate`。
4. 可选：加一个散点图观察费用关系：
   - X 轴：`MonthlyCharges`
   - Y 轴：`TotalCharges`
   - 图例：`Churn`

## 6. 联动与校验

1. 在任意页点击某个筛选项，检查其他图是否联动变化。
2. 新建一个卡片放 `Churn No`，验证：
   - `Churn Yes + Churn No = Total Rows`
3. 在不筛选时，`Churn Rate` 应约为 `22.52%`。

## 7. 保存文件

1. 点击 `文件` -> `另存为`。
2. 保存到：`D:/workplace/kaggle/PS-S6E3/`
3. 文件名：`PS-S6E3_EDA.pbix`

## 8. 常见问题

1. DAX 报红线：
   - 先检查引号是否是英文半角 `"`。
   - 检查表名是否确实是 `train`。
2. 看不到“新建度量值”：
   - 右键的是字段不是表；请右键 `train` 表名。
3. 图表为空：
   - 检查是否被切片器筛到无数据；点击橡皮擦清空切片器。
