# `group_cols` 多列分组语义（分组键 / 输出保留列 / 权重表主键 / 排序）

本文件用于定义统一 TOPSIS 在 `group_cols` 支持 2+ 多列时的分组语义与输出约定，为后续实现与验收对齐提供依据。

相关前置约定：
- 参数输入格式见 `concepts_group_cols_year_col.md`
- 分支判定与同名去重见 `concepts_group_cols_year_col.md` 的“判定规则（分支选择 + 同名去重）”

## 1) 分组键（Group Key）

### 1.1 联合分组（joint grouping）

- `group_cols` 支持 1/N 列；当 N>=2 时，按 **多列联合分组**：
  - 分组键为按顺序取值的元组：`(g1, g2, ..., gN)`；
  - 语义等价于 `df.groupby(group_cols, ...)`（一次 groupby，非嵌套循环）。

### 1.2 列顺序与去重

- 分组键列顺序：严格使用归一化后的 `group_cols` 顺序（stable unique 后的顺序）。
- 同名去重：若 `year_col` 与 `group_cols` 同名，分组键列不包含该列（详见判定规则文档）。
- 不生成拼接字符串键：分组键在输出中以 **多列原列名** 形式保留，不额外引入 `group_key` 之类的合成列。

### 1.3 缺失值（NA）处理

- 分组键列出现缺失值（NA/NaN）时，**不应静默丢行**。
- 实现优先：`groupby(..., dropna=False)`（将 NA 视为一个合法组）；若运行环境不支持，则需先对分组键列做可逆的缺失值填充后再分组（输出前可选择性还原）。

## 2) 输出表保留列（Entropy/TOPSIS）

当按 `group_cols` 分组计算时：

- `group_cols` 各列必须作为普通列在 `熵权`/`topsis` 输出中保留（不可作为 index 丢失）。
- 若提供 `year_col`，则 `year_col` **仅作为 ID 列保留**（不参与分组键、不参与指标计算、且不重复）。
- `id_cols`（若存在）作为额外 ID 列保留；与 `year_col`/`group_cols` 重复的列需要去重，避免输出出现重复列名。

> 备注：ID 列最终顺序与权重输出规范见 `output_spec_entropy_topsis.md`（`year_col` → `group_cols` → `id_cols`，stable unique 去重）。

## 3) 权重表（Weights Table）

按 `group_cols` 分组计算时，权重输出采用“一组一行”：

- 行粒度：每个分组键（联合键）对应 1 行权重。
- 主键（Primary Key）：`group_cols` 各列组成复合主键（N 列联合唯一）。
- 列结构：`[*group_cols, *metric_cols]`  
  - `group_cols` 放在最前，按输入顺序排列；
  - 指标权重列顺序与指标列顺序一致（由实现侧在计算时确定并保持稳定）。

## 4) 排序（Ordering）

为保证可复现与对齐验收，统一排序策略：

- 组顺（group order）：按 `group_cols` 的联合键升序排序（与 pandas `groupby(sort=True)` 一致的字典序）。
- 组内顺序：保持原始数据在该组内的行顺序不变。
- `weights` 表行顺序：与组顺一致（按联合键升序）。
