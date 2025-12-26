from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Default negative indicators inherited from existing scripts.
DEFAULT_NEGATIVE_INDICATORS = {
    "服务业二氧化硫排放量/服务业增加值 二氧化硫/服务业增加值",
    "服务业污水排放量/服务业增加值 废水排放/服务业增加值",
}


def parse_negative_indicators(text: Optional[str]) -> set[str]:
    """Parse a comma/newline separated negative-indicator list."""
    if not text:
        return {item.strip() for item in DEFAULT_NEGATIVE_INDICATORS if item.strip()}
    parts: list[str] = []
    for chunk in text.replace("\n", ",").split(","):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return set(parts)


@dataclass(frozen=True)
class EntropyTopsisResult:
    """Entropy-weight TOPSIS output tables.

    - entropy: '熵权' sheet table
    - topsis: 'topsis' sheet table
    - weights: entropy weights (per group if grouped)
    - metric_cols: metric columns used in computation
    """

    entropy: pd.DataFrame
    topsis: pd.DataFrame
    weights: pd.DataFrame
    metric_cols: list[str]


def _normalize_minmax(values: np.ndarray, *, is_negative: bool) -> np.ndarray:
    s_max = np.nanmax(values)
    s_min = np.nanmin(values)
    if np.isclose(s_max, s_min):
        return np.zeros_like(values, dtype=float)
    if is_negative:
        return (s_max - values) / (s_max - s_min)
    return (values - s_min) / (s_max - s_min)


def _entropy_weights(standardized_values: pd.DataFrame) -> np.ndarray:
    """Compute entropy weights (g_j) following the existing '单省份' script logic."""
    col_sums = standardized_values.sum(axis=0).replace(0, np.nan)
    p = standardized_values.div(col_sums, axis=1).fillna(0.0)

    m = len(p)
    n = len(p.columns)
    k = 0.0 if m <= 1 else 1.0 / np.log(m)

    e_list: list[float] = []
    for col in p.columns:
        p_col = p[col].to_numpy(dtype=float)
        mask = p_col > 0
        if mask.sum() == 0 or k == 0.0:
            e = 0.0
        else:
            e = -k * float(np.sum(p_col[mask] * np.log(p_col[mask])))
        e_list.append(e)

    e_j = np.array(e_list, dtype=float)
    one_minus_e = 1.0 - e_j
    den = float(one_minus_e.sum())
    if np.isclose(den, 0.0) or n == 0:
        return np.full(n, 1.0 / max(n, 1), dtype=float)
    return one_minus_e / den


def _topsis_scores(standardized_values: pd.DataFrame, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = standardized_values.to_numpy(dtype=float)
    m, n = x.shape
    x_prime = np.zeros((m, n), dtype=float)
    for j in range(n):
        col = x[:, j]
        norm = float(np.linalg.norm(col))
        if np.isclose(norm, 0.0):
            x_prime[:, j] = 0.0
        else:
            x_prime[:, j] = col / norm

    weighted = x_prime * weights
    p_plus = np.max(weighted, axis=0)
    p_minus = np.min(weighted, axis=0)
    d_plus = np.sqrt(np.sum((weighted - p_plus) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - p_minus) ** 2, axis=1))
    den = d_plus + d_minus
    c = np.zeros_like(d_minus)
    mask = ~np.isclose(den, 0.0)
    c[mask] = d_minus[mask] / den[mask]
    rank = pd.Series(c).rank(ascending=False, method="min").astype(int).to_numpy()
    return d_plus, d_minus, c, rank


def append_weights_row(
    entropy_df: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    weights: pd.Series,
    label: str = "权重",
) -> pd.DataFrame:
    """Append a '权重' row to the entropy table (legacy-friendly output)."""
    row = {col: "" for col in entropy_df.columns}
    if id_cols:
        row[str(id_cols[0])] = label
    for col_name, weight in weights.items():
        if col_name in entropy_df.columns:
            row[col_name] = float(weight)
    return pd.concat([entropy_df, pd.DataFrame([row])], ignore_index=True)


def run_entropy_topsis(
    df: pd.DataFrame,
    *,
    id_cols: Optional[Sequence[str]] = None,
    negative_cols: Optional[Iterable[str]] = None,
    eps_shift: float = 0.01,
    group_cols: Optional[Sequence[str]] = None,
    year_col: Optional[str] = None,
    **kwargs,
) -> EntropyTopsisResult:
    """Run entropy-weight TOPSIS with optional grouping.

    Args:
        df: Input DataFrame with metrics and optional ID columns.
        id_cols: ID columns that are not used in metric calculation.
        negative_cols: Set of column names that are negative indicators.
        eps_shift: Non-negative shift constant for standardized values.
            - Default (0.01): Adds 0.01 to normalized values to avoid zeros
              before entropy weight calculation.
            - 0: Disables shift; uses normalized values directly.
            - Positive value (> 0): Shifts normalized values by the specified amount.
        group_cols: Grouping column names (multi-column supported).
        year_col: Year column name (used as ID column and can combine with grouping).
        **kwargs: Backward compatibility for negative indicators:
            - negative_indicators, neg_cols, cost_cols (aliases for negative_cols)

    Behavior for eps_shift:
        - eps_shift=0: Normalized values are used as-is for entropy calculation.
          Useful when data is already non-negative after normalization.
        - eps_shift>0: Added to normalized values (range [0,1]) before entropy
          calculation to prevent log(0) errors. A common default is 0.01.
        - eps_shift must be non-negative; negative values are not allowed.

    Branch logic:
        - 仅分组: Only grouping, no year column
        - 仅年份: Only year column, no grouping
        - 分组+年份: Both grouping and year column
        - 同名去重: Handle when group_cols and year_col have the same name
    """
    if df is None or df.empty:
        raise ValueError("输入数据为空，无法计算。")

    # Handle backward compatibility for negative indicators
    if negative_cols is None:
        negative_cols = (
            kwargs.get("negative_indicators")
            or kwargs.get("neg_cols")
            or kwargs.get("cost_cols")
        )

    # If it's a string, parse it; if it's None, use default from parse_negative_indicators
    if isinstance(negative_cols, str):
        cleaned_negative = parse_negative_indicators(negative_cols)
    elif negative_cols is None:
        cleaned_negative = parse_negative_indicators(None)
    else:
        # It's an iterable of strings
        cleaned_negative = {str(x).strip() for x in negative_cols if str(x).strip()}

    work_df = df.copy()
    work_df.columns = work_df.columns.map(lambda x: str(x).strip())

    cleaned_id_cols = [str(c).strip() for c in (id_cols or []) if str(c).strip()]

    # Normalize group_cols: split by comma/newline, strip, dedup, keep order
    def _normalize_group_cols(group_cols_input: Optional[Sequence[str]]) -> list[str]:
        if not group_cols_input:
            return []
        flat_parts: list[str] = []
        for item in group_cols_input:
            for chunk in str(item).replace("\n", ",").split(","):
                cleaned = chunk.strip()
                if cleaned:
                    flat_parts.append(cleaned)
        seen = set()
        result = []
        for item in flat_parts:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    normalized_group_cols = _normalize_group_cols(group_cols)
    year_col_clean = str(year_col).strip() if year_col else ""

    # 同名去重: Remove year_col from group_cols if present
    if year_col_clean and year_col_clean in normalized_group_cols:
        normalized_group_cols = [col for col in normalized_group_cols if col != year_col_clean]

    # Determine final ID column order: year_col -> group_cols -> id_cols
    final_id_cols: list[str] = []
    if year_col_clean:
        final_id_cols.append(year_col_clean)
    if normalized_group_cols:
        final_id_cols.extend(normalized_group_cols)

    # Add remaining id_cols (stable unique, dedup against existing)
    seen = set(final_id_cols)
    for col in cleaned_id_cols:
        if col not in seen:
            final_id_cols.append(col)
            seen.add(col)

    # Verify all final ID columns exist
    for col in final_id_cols:
        if col not in work_df.columns:
            raise ValueError(f"ID 列不存在：{col}")

    def _select_metric_cols(exclude_cols: set[str]) -> list[str]:
        metric_cols = [c for c in work_df.columns if c not in exclude_cols]
        if not metric_cols:
            raise ValueError("未找到可用于计算的指标列。")
        numeric_raw = work_df[metric_cols].apply(pd.to_numeric, errors="coerce")
        all_nan_cols = [c for c in metric_cols if numeric_raw[c].isna().all()]
        usable_cols = [c for c in metric_cols if c not in set(all_nan_cols)]
        if not usable_cols:
            raise ValueError("所有指标列都无法转换为数值，无法计算。")
        return usable_cols

    def _run_single(table_df: pd.DataFrame, metric_cols: Sequence[str], is_grouped: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        numeric_raw = table_df[list(metric_cols)].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_raw.fillna(numeric_raw.mean()).fillna(0.0)

        normalized = pd.DataFrame(index=table_df.index)
        for col in metric_cols:
            is_negative = col in cleaned_negative
            normalized[col] = _normalize_minmax(numeric_df[col].to_numpy(dtype=float), is_negative=is_negative)

        standardized = normalized + float(eps_shift)
        weights_arr = _entropy_weights(standardized)

        d_plus, d_minus, c, rank = _topsis_scores(standardized, weights_arr)

        weights_series = pd.Series(weights_arr, index=metric_cols, name="weight")

        contributions = standardized.multiply(weights_arr, axis=1)
        entropy_table = contributions.copy()
        entropy_table["综合指数"] = contributions.sum(axis=1)

        # For entropy table, include proper ID columns based on context
        if is_grouped:
            # In grouped mode, include group_cols and year_col (if exists)
            id_cols_for_output = []
            if year_col_clean:
                id_cols_for_output.append(year_col_clean)
            id_cols_for_output.extend(normalized_group_cols)
            # Add other id_cols that aren't already included
            for col in cleaned_id_cols:
                if col not in id_cols_for_output:
                    id_cols_for_output.append(col)
        else:
            # In non-grouped mode, use final_id_cols
            id_cols_for_output = final_id_cols

        if id_cols_for_output:
            entropy_table = pd.concat([table_df[id_cols_for_output], entropy_table], axis=1)
        entropy_table = entropy_table.reset_index(drop=True)

        topsis_table = pd.DataFrame(
            {
                "正理想解距离D+": d_plus,
                "负理想解距离D-": d_minus,
                "相对接近度C": c,
                "排名": rank,
            }
        )
        if id_cols_for_output:
            topsis_table = pd.concat([table_df[id_cols_for_output].reset_index(drop=True), topsis_table], axis=1)
        topsis_table = topsis_table.reset_index(drop=True)

        return entropy_table, topsis_table, weights_series

    # Branch logic: Check if grouping is enabled (group_cols not empty)
    if not normalized_group_cols:
        # No grouping (either "仅年份" or "不分组不按年份")
        metric_cols = _select_metric_cols(set(final_id_cols))
        entropy_table, topsis_table, weights_series = _run_single(work_df, metric_cols, is_grouped=False)
        weights_df = pd.DataFrame([weights_series.to_dict()], columns=metric_cols)
        return EntropyTopsisResult(entropy=entropy_table, topsis=topsis_table, weights=weights_df, metric_cols=metric_cols)

    # Grouping enabled (either "仅分组" or "分组+年份")
    entropy_tables: list[pd.DataFrame] = []
    topsis_tables: list[pd.DataFrame] = []
    weights_rows: list[dict[str, object]] = []
    exclude_cols = set(normalized_group_cols)
    if year_col_clean:
        exclude_cols.add(year_col_clean)
    exclude_cols.update(cleaned_id_cols)
    all_metric_cols = _select_metric_cols(exclude_cols)

    # Group by normalized_group_cols (joint grouping)
    grouped = work_df.groupby(normalized_group_cols, sort=True, dropna=False)
    for group_key, group_df in grouped:
        if isinstance(group_key, tuple):
            group_key_tuple = group_key
        else:
            group_key_tuple = (group_key,)

        entropy_table, topsis_table, weights_series = _run_single(group_df, all_metric_cols, is_grouped=True)

        # Create weights row with group key columns
        weights_row: dict[str, object] = {}
        for i, group_col in enumerate(normalized_group_cols):
            weights_row[group_col] = group_key_tuple[i]

        for col in all_metric_cols:
            weights_row[col] = float(weights_series[col])
        weights_rows.append(weights_row)
        entropy_tables.append(entropy_table)
        topsis_tables.append(topsis_table)

    merged_entropy = pd.concat(entropy_tables, axis=0, ignore_index=True) if entropy_tables else pd.DataFrame()
    merged_topsis = pd.concat(topsis_tables, axis=0, ignore_index=True) if topsis_tables else pd.DataFrame()
    weights_df = pd.DataFrame(weights_rows)

    # Ensure proper column ordering for weights table: [*group_cols, *metric_cols]
    if normalized_group_cols and all_metric_cols:
        weights_df = weights_df[[*normalized_group_cols, *all_metric_cols]]

    return EntropyTopsisResult(entropy=merged_entropy, topsis=merged_topsis, weights=weights_df, metric_cols=all_metric_cols)
