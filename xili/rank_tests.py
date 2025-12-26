from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def run_rank_tests(
    df: pd.DataFrame,
    *,
    mode: Literal["questions", "group"],
    question_cols: Optional[list[str]] = None,
    dv_col: Optional[str] = None,
    group_col: Optional[str] = None,
    group_cols: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run non-parametric rank tests (Mann-Whitney U / Kruskal-Wallis H).

    Parameters
    - mode="questions": iterate question columns, test dv_col across groups of each question.
    - mode="group": iterate all columns except group_cols, test values across group_cols groups.
                   Supports both single group_col and multiple group_cols.

    统计方法 / Methods
    - 两组：Mann-Whitney U（非参数、独立样本秩和检验）
    - 多组：Kruskal-Wallis H（非参数，一元 ANOVA 替代）
    """
    if df is None or df.empty:
        raise ValueError("输入数据为空，无法进行检验。")

    if mode == "questions":
        if not question_cols:
            raise ValueError("mode='questions' 时必须提供 question_cols。")
        if not dv_col:
            raise ValueError("mode='questions' 时必须提供 dv_col。")
        if dv_col not in df.columns:
            raise ValueError(f"因变量列不存在：{dv_col}")

        rows: list[dict[str, object]] = []
        for q_col in question_cols:
            if q_col not in df.columns:
                raise ValueError(f"自变量列不存在：{q_col}")

            unique_vals = sorted(df[q_col].dropna().unique())
            n_groups = len(unique_vals)

            if n_groups < 2:
                rows.append(
                    {
                        "Question": q_col,
                        "TestUsed": "NoTest(only1group)",
                        "Statistic": None,
                        "pValue": None,
                        "Significant": False,
                        "Groups": n_groups,
                    }
                )
                continue

            groups: list[pd.Series] = []
            empty_groups: list[object] = []
            for val in unique_vals:
                group_data = _to_numeric(df.loc[df[q_col] == val, dv_col]).dropna()
                if group_data.empty:
                    empty_groups.append(val)
                groups.append(group_data)

            if empty_groups:
                rows.append(
                    {
                        "Question": q_col,
                        "TestUsed": f"NoTest(empty_group:{empty_groups})",
                        "Statistic": None,
                        "pValue": None,
                        "Significant": False,
                        "Groups": n_groups,
                    }
                )
                continue

            try:
                if n_groups == 2:
                    stat, p_val = mannwhitneyu(groups[0].values, groups[1].values, alternative="two-sided")
                    test_used = "Mann-Whitney U"
                else:
                    stat, p_val = kruskal(*[g.values for g in groups])
                    test_used = "Kruskal-Wallis"
            except Exception as exc:  # noqa: BLE001 - keep CLI robust
                rows.append(
                    {
                        "Question": q_col,
                        "TestUsed": f"NoTest(error:{type(exc).__name__})",
                        "Statistic": None,
                        "pValue": None,
                        "Significant": False,
                        "Groups": n_groups,
                    }
                )
                continue

            rows.append(
                {
                    "Question": q_col,
                    "TestUsed": test_used,
                    "Statistic": stat,
                    "pValue": p_val,
                    "Significant": bool(p_val < alpha),
                    "Groups": n_groups,
                }
            )

        return pd.DataFrame(rows)

    if mode == "group":
        # Handle group_cols or single group_col for backward compatibility
        if group_cols:
            # Multi-column group mode
            for col in group_cols:
                if col not in df.columns:
                    raise ValueError(f"分组列不存在：{col}")
            groups_used = group_cols
        elif group_col:
            # Single-column group mode (backward compatibility)
            if group_col not in df.columns:
                raise ValueError(f"分组列不存在：{group_col}")
            groups_used = [group_col]
        else:
            raise ValueError("mode='group' 时必须提供 group_col 或 group_cols。")

        # Create grouped data using pandas groupby
        grouped = df.groupby(groups_used, sort=True, dropna=False)
        unique_group_keys = list(grouped.groups.keys())
        n_groups = len(unique_group_keys)
        test_columns = [col for col in df.columns if col not in groups_used]

        rows: list[dict[str, object]] = []
        for col in test_columns:
            if n_groups < 2:
                rows.append(
                    {
                        "变量": col,
                        "检验方法": "NoTest(only1group)",
                        "statistic": None,
                        "p值": None,
                    }
                )
                continue

            groups: list[pd.Series] = []
            empty_groups: list[object] = []
            for group_key in unique_group_keys:
                # Get the group data using the groupby object
                group_values = _to_numeric(grouped.get_group(group_key)[col]).dropna()
                if group_values.empty:
                    empty_groups.append(group_key)
                groups.append(group_values)

            if empty_groups:
                rows.append(
                    {
                        "变量": col,
                        "检验方法": f"NoTest(empty_group:{empty_groups})",
                        "statistic": None,
                        "p值": None,
                    }
                )
                continue

            try:
                if n_groups == 2:
                    stat, p_val = mannwhitneyu(groups[0].values, groups[1].values, alternative="two-sided")
                    method = "Mann-Whitney U检验"
                else:
                    stat, p_val = kruskal(*[g.values for g in groups])
                    method = "Kruskal-Wallis H检验"
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "变量": col,
                        "检验方法": f"NoTest(error:{type(exc).__name__})",
                        "statistic": None,
                        "p值": None,
                    }
                )
                continue

            rows.append(
                {
                    "变量": col,
                    "检验方法": method,
                    "statistic": stat,
                    "p值": p_val,
                }
            )

        return pd.DataFrame(rows)

    raise ValueError(f"未知 mode：{mode}")

