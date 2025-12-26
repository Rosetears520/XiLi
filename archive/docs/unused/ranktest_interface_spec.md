# RankTest Unified Interface Spec (`mode=questions|group` / `group_cols` multi)

This document defines the unified interface contract for RankTest so the unified CLI/GUI and legacy wrappers can be aligned while adding multi-column `group_cols`.

References:
- Legacy baseline: `legacy_scripts_baseline.md`
- `group_cols` input format + dedup: `concepts_group_cols_year_col.md`
- Multi-column grouping semantics (ordering/NA): `semantics_multi_group_cols.md`

---

## 1) Modes / 模式

- `mode=questions`：对每个 `question_col`，按该列的不同取值分组，对 `dv_col` 做两组/多组秩和检验。
- `mode=group`：按 `group_cols`（支持 2+ 多列联合分组）划分组别，对除分组列外的各变量逐列做两组/多组秩和检验。

检验方法（两模式一致）：
- 两组：Mann-Whitney U（非参数、独立样本秩和检验）
- 多组：Kruskal-Wallis H（非参数，一元 ANOVA 替代）

## 2) Parameters / 参数集合

### 2.1 Common（两模式共用）

- `input_file`：必填
- `output_file`：可选（不同入口/旧脚本默认文件名不同；见基线文档）
- `alpha`：可选，默认 `0.05`
- `run_dir`：可选（同仓库其它脚本：`runs/<timestamp_uuid>/`）

### 2.2 `mode=questions`

- `question_cols`：必填；逗号/换行分隔；逐项 `strip()`；按出现顺序去重（stable unique）
- `dv_col`：可选；默认 `BIU`（与旧脚本一致）
- 互斥：不应接受 `group_cols`（若提供应报错）

### 2.3 `mode=group`

- `group_cols`：可选；默认为 `区域`（兼容旧脚本默认值）；逗号/换行分隔；逐项 `strip()`；按出现顺序去重（stable unique）
- 兼容别名：`group_col` 等价于 `group_cols` 的单列输入
- 互斥：不应接受 `question_cols`（若提供应报错）
- `dv_col`：忽略（该模式不使用）

### 2.4 Parsing / 解析规则（所有入口一致）

- 列名统一 `strip()`（去前后空格）
- `question_cols` / `group_cols`：按逗号/换行拆分，过滤空项，按出现顺序去重
- 若同时提供 `group_col` 与 `group_cols`：
  - 以 `group_cols` 为准（`group_col` 仅作为兼容别名；CLI 应给出弃用提示）

## 3) Group construction for `mode=group`（`group_cols` 支持 2+）

- 分组键：按归一化后的 `group_cols` 顺序取值构造联合键 `(g1, g2, ..., gN)`。
- 等价实现：`df.groupby(group_cols, sort=True, dropna=False)`（一次 groupby，非嵌套循环）。
- 缺失值（NA）：分组键列出现 NA/NaN 时 **不得静默丢行**；NA 应作为一个合法组别参与检验。
- 不生成拼接字符串键：不额外引入 `group_key` 之类的合成列。

组顺（group order）：
- 组别顺序应可复现：按联合键字典序升序（与 `groupby(sort=True)` 一致）；详见 `semantics_multi_group_cols.md`。

## 4) Test behavior / 检验逻辑

两模式共享规则：
- 组数 < 2：不做检验（NoTest）
- 组数 = 2：Mann-Whitney U（two-sided）
- 组数 >= 3：Kruskal-Wallis H

### 4.1 `mode=questions`

- 对每个 `question_col`：按该列的不同取值分组（取值集合不包含 NA；为与旧实现一致）。
- 每组取 `dv_col`，转换为数值（`to_numeric(errors="coerce")`）后 `dropna()`。
- 若任一组在数值化后为空，则该题目不做检验（兼容旧实现的“保守失败”策略）。

### 4.2 `mode=group`

- 被检验变量列：默认遍历除 `group_cols` 之外的所有列（兼容旧脚本行为）。
- 对每个变量列：对每个组取该列数据，数值化后 `dropna()`。
- 若任一组在数值化后为空，则该变量不做检验（兼容旧实现的“保守失败”策略）。

## 5) Output schema / 输出列（当前约定）

- `mode=questions` 输出列：`Question`, `TestUsed`, `Statistic`, `pValue`, `Significant`, `Groups`
- `mode=group` 输出列：`变量`, `检验方法`, `statistic`, `p值`

