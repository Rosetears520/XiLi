# group_cols / year_col 统一约定（输入格式）

本文件用于定义后续统一 CLI/GUI 的两个核心概念：
- `year_col`：年份列（单列）
- `group_cols`：分组列（多列，支持 2+）

## 1) 命名

- 代码/文档统一使用 `year_col`（单数）与 `group_cols`（复数）两个名称。
- CLI 参数推荐：`--year-col` 与 `--group-cols`。

## 2) 输入格式（CLI/GUI 一致）

### year_col

- 类型：字符串（列名）。
- 示例：`--year-col 年份`。
- 值格式建议：Excel 中可为 int/float/str，但推荐使用 4 位年份（如 2023）；该列只作为 ID 保留，不参与指标计算。

### group_cols

- 类型：字符串列表（列名），允许 2+ 列。
- 解析规则：逗号/换行分隔，逐项 `strip()`，过滤空项，按出现顺序去重。
- 示例：`--group-cols 省份,行业` 或 `--group-cols 省份
行业`。
- 兼容：旧参数 `--group-col` 等价于 `--group-cols` 仅 1 列（后续任务会统一）。



## 3) 判定规则（分支选择 + 同名去重）

输入归一化（所有入口一致）：
- `year_col`：`strip()` 后为空视为未提供。
- `group_cols`：先按逗号/换行拆分，再逐项 `strip()`、过滤空项、按出现顺序去重。
- 同名去重：若 `year_col` 出现在 `group_cols` 中，则从 `group_cols` 移除该项（避免重复 ID/分组列）。

分支选择（基于归一化后的值）：
- `group_cols` 为空 且 `year_col` 为空：不分组、不按年份特殊处理（全表一次）。
- `group_cols` 为空 且 `year_col` 非空：仅年份分支（`year_col` 作为 ID 列保留）。
- `group_cols` 非空 且 `year_col` 为空：仅分组分支（按 `group_cols` 联合分组）。
- `group_cols` 非空 且 `year_col` 非空：分组+年份分支（按 `group_cols` 联合分组；`year_col` 仅作为 ID 列保留）。

等价分支：
- 若用户同时传入 `--year-col` 和 `--group-cols`，但去除同名后 `group_cols` 变为空，则等价于“仅年份分支”。


## 4) 备注

- 列名统一做 `strip()`（去前后空格）。
- `group_cols` 多列分组语义（分组键/输出保留列/权重表主键/排序）见 `semantics_multi_group_cols.md`。
- 输出（ID 列顺序 / 权重输出 / 追加权重行）规范见 `output_spec_entropy_topsis.md`。
- 本文件的“分支选择 + 同名去重”作为统一规范；代码落地与输出对齐见 TODO 的 B/C 项（`xili/entropy_topsis.py`/`xili/cli.py`/GUI）。
