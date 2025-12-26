# RankTest interface spec

> NOTE: The source of truth is `ranktest_interface_spec.md`. This file is kept as a pointer and may be removed later.

This document defines the unified RankTest interface (`mode=questions|group`) and the multi-column `group_cols`
group construction rules. It is the contract for future CLI/GUI and legacy-wrapper alignment.

---

# RankTest 统一接口规范（`mode=questions|group` + `group_cols` 多列分组）

本文件用于定义统一秩和检验（RankTest）的接口参数集合与行为，作为后续 CLI/GUI/旧脚本 wrapper 合并的依据。

统计方法（与旧脚本保持一致）：
- 两组：Mann-Whitney U（非参数、独立样本秩和检验）
- 多组：Kruskal-Wallis H（非参数，一元 ANOVA 替代）

相关前置约定：
- `group_cols` 的输入格式、去空白、去重规则见 `concepts_group_cols_year_col.md`


## 1) 统一入口与模式

统一入口（推荐）：`python -m xili.cli ranktest ...`

模式（必填，二选一）：
- `mode=questions`：按题目列（自变量）分组，对因变量列做秩和检验并输出题目级汇总。
- `mode=group`：按分组列（`group_cols`，支持多列联合分组）分组，遍历其余列做秩和检验并输出变量级结果。


## 2) 参数集合（CLI/GUI 一致）

### 2.1 通用参数（两种模式都支持）

- `input_file`：必填，Excel 输入文件路径。
- `output_file`：可选，输出 Excel 文件路径（统一 CLI 的默认文件名由实现侧决定；旧脚本 wrapper 需保持各自默认值不变）。
- `alpha`：可选，显著性水平（默认 `0.05`；旧脚本内固定为 `0.05`）。
- `run_dir`：可选，运行目录（默认 `runs/<timestamp_uuid>/`）。

### 2.2 questions 模式参数

- `mode=questions`：必填。
- `question_cols`：必填，自变量列名列表（逗号/换行分隔，逐项 `strip()`，过滤空项）。
- `dv_col`：可选，因变量列名（默认 `BIU`，与旧脚本一致）。

约束：
- questions 模式不接收 `group_cols`（若提供应报错/提示忽略，具体以实现侧为准；推荐报错以避免误用）。

输出（与旧脚本一致）：
- Sheet：`Summary`
- 列：`Question`、`TestUsed`、`Statistic`、`pValue`、`Significant`、`Groups`

### 2.3 group 模式参数

- `mode=group`：必填。
- `group_cols`：必填，分组列名列表（支持 1/N，多列为联合分组）。
  - 兼容：旧参数名 `group_col` 等价于 `group_cols` 仅 1 列（后续实现需保留别名）。

约束：
- group 模式不接收 `question_cols`/`dv_col`（若提供应报错）。

输出（保持旧脚本列名风格）：
- 列：`变量`、`检验方法`、`statistic`、`p值`


## 3) `group_cols` 多列分组：组别构造规则（核心）

当 `mode=group` 且 `len(group_cols) >= 1` 时，组别（Groups）按 **多列联合分组** 构造：

### 3.1 联合分组（joint grouping）

- 组别键（group key）为按顺序取值的元组：`(g1, g2, ..., gN)`；
- 语义等价于一次性 `df.groupby(group_cols, ...)`（一次 groupby，非嵌套循环）。

### 3.2 列顺序与去重

- `group_cols` 的列顺序：严格使用归一化后的输入顺序（stable unique）。
- 不生成拼接字符串键：不额外引入 `group_key`/`组别` 等合成列作为分组依据（内部可用元组表示）。

### 3.3 缺失值（NA）处理

- `group_cols` 中任意列出现缺失值（NA/NaN）时，不应静默丢行。
- 推荐实现：`groupby(..., dropna=False)`，将 NA 作为合法组别参与分组与检验。

### 3.4 组别数量与可检验性

- 组别数量 `K` 为联合键的去重后个数。
- 若 `K < 2`：无法进行两组/多组检验，应返回 `NoTest(only1group)`（或等价标识）。
- 若存在空组（某组在待检验变量列上全为缺失/无法转数值）：应返回 `NoTest(empty_group:...)`（或等价标识），避免异常中断。


## 4) CLI 示例（未来统一接口）

```bash
# questions 模式：对每个题目列分组检验 dv_col
python -m xili.cli ranktest --mode questions --input-file data/input.xlsx --question-cols Q1,Q2,Q3 --dv-col BIU --alpha 0.05 --output-file rank_tests.xlsx

# group 模式：多列联合分组（区域 + 性别），遍历其余列做检验
python -m xili.cli ranktest --mode group --input-file data/input.xlsx --group-cols 区域,性别 --alpha 0.05 --output-file rank_tests.xlsx
```
