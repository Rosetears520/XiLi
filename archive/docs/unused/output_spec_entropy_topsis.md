# Entropy TOPSIS Output Spec (ID Order / Dedup / Weights)

This file is the source of truth for the exact table layout: final ID column order, dedup rules, and how weights are represented (either as a separate weights table or as a legacy-appended row). Keep wrappers and `xili/*` implementations aligned with this contract.

This document defines the output contract for Entropy-weight TOPSIS (`熵权`/`topsis`/`weights`) so the unified CLI/GUI and legacy wrappers can be aligned and regression-checked.

References:
- Concepts + branching: `concepts_group_cols_year_col.md`
- Multi-column grouping semantics + ordering: `semantics_multi_group_cols.md`
- Legacy baseline for compatibility: `legacy_scripts_baseline.md`

---

## 1) Terms / 术语

- `year_col`：年份列（单列）。只作为 ID 保留，不参与指标计算。
- `group_cols`：分组列（0/1/N，支持 2+）。用于“按组计算权重/得分”的组别构造（联合分组）。
- `id_cols`：额外需要保留的 ID 列（不参与指标计算）。
- `metric_cols`：参与计算的指标列（数值列；会剔除不可用列）。

## 2) Final ID column order (Entropy/TOPSIS shared)

### 2.1 Normalization + dedup (input side)

All entrypoints MUST normalize inputs consistently (see `concepts_group_cols_year_col.md`):
- Column names: `strip()`
- `group_cols`: split by comma/newline, `strip()` each item, drop empties, stable-unique in first-seen order
- Same-name dedup: if `year_col` appears in `group_cols`, remove it from `group_cols`

### 2.2 Output ID column order (output side)

The final ID column order is fixed as:

1. `year_col` (if provided and non-empty)
2. `group_cols` (normalized order)
3. Remaining `id_cols` (after removing duplicates with 1/2; stable-unique)

Dedup policy:
- Stable unique: keep the first occurrence, drop later duplicates
- Outputs MUST NOT contain duplicated column names

> Rationale: prioritize legacy compatibility (typical baseline order is `年份` first, `省份` second).

## 3) `熵权` table (Entropy sheet)

- Sheet name: `熵权`
- Column order:
  - `[*final_id_cols, *metric_cols(contribution), 综合指数]`
- Notes:
  - Each `metric_cols` output column is the contribution value: (standardized value) * (weight), consistent with legacy scripts.
  - `综合指数` is the row-wise sum of contributions.

## 4) `topsis` table (TOPSIS sheet)

- Sheet name: `topsis`
- Column order:
  - `[*final_id_cols, 正理想解距离D+, 负理想解距离D-, 相对接近度C, 排名]`

## 5) Weights output (`weights`)

### 5.1 Standard `weights` table shape (recommended)

The `weights` table is the normalized way to expose weights for debugging, acceptance checks, and multi-group support.

- Not grouped (`group_cols` empty):
  - Row granularity: single row (run once on the full table)
  - Columns: `[*metric_cols]`
- Grouped (`group_cols` non-empty; both “group-only” and “group+year” branches):
  - Row granularity: one row per joint group
  - Primary key: `group_cols` (composite key)
  - Columns: `[*group_cols, *metric_cols]`
  - Important: `year_col` is NOT part of the weights primary key (it stays as a detail-level ID only)

Ordering follows `semantics_multi_group_cols.md`.

### 5.2 Legacy-style appended “weights row” (compat)

Some legacy scripts append weights into the `熵权` sheet as an extra row (non-standard output).

Spec:
- Only applicable to NOT-grouped runs (single global weight vector).
- Append exactly 1 row to the end of `熵权`:
  - Put literal `权重` in the 1st ID column
  - Other ID columns are empty strings
  - Metric columns contain weights
  - `综合指数` is left empty

Grouped runs MUST NOT append weights rows by default (legacy scripts also do not).

## 6) Legacy wrapper compatibility requirements

Legacy wrappers must match `legacy_scripts_baseline.md`:

- Excel output stays at 2 sheets: `熵权` and `topsis` (do NOT add a `weights` sheet in wrappers).
- `熵权topsis法-单省份.py` and `熵权topsis法-年份.py`: append the weights row (as in 5.2).
- `熵权topsis法---加入省份列.py` and `熵权topsis法 - 以省份为分组 -年份.py`: do not append weights row.
- Keep baseline ID column order (typical: `年份` then `省份`).
