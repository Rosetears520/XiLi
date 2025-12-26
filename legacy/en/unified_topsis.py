#!/usr/bin/env python3
"""
Unified TOPSIS CLI Script

This script provides a unified interface for entropy-weight TOPSIS calculation,
replacing the 4 legacy scripts with a single comprehensive command-line interface.

Supported legacy script equivalents:
1. 熵权topsis法 - 以省份为分组 -年份.py -> --group-cols 省份 --year-col 年份
2. 熵权topsis法---加入省份列.py -> --id-cols 年份,省份
3. 熵权topsis法-单省份.py -> --id-cols 学校名称
4. 熵权topsis法-年份.py -> --id-cols 年份

Branch logic:
- No group_cols, no year_col: Process entire table as one group
- No group_cols, year_col provided: Process entire table with year_col as ID column
- group_cols provided, no year_col: Group by group_cols
- group_cols and year_col provided: Group by group_cols, year_col as ID column
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, List, Set

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path
from xili.entropy_topsis import parse_negative_indicators, run_entropy_topsis, append_weights_row


def _parse_csv_list(text: Optional[str]) -> List[str]:
    """解析逗号或换行分隔的列表，去除空格、过滤空项并保持顺序去重。"""
    if not text:
        return []

    parts = re.split(r"[,\n]", text)
    seen = set()
    result = []
    for p in parts:
        s = p.strip()
        if s and s not in seen:
            result.append(s)
            seen.add(s)
    return result


def _normalize_inputs(
    year_col: Optional[str],
    group_cols: List[str],
    id_cols: List[str],
    df_columns: List[str],
) -> tuple[Optional[str], List[str], List[str]]:
    """
    Normalize inputs according to concepts_group_cols_year_col.md

    Returns:
        - Normalized year_col (stripped, or None if empty)
        - Normalized group_cols (deduplicated, with year_col removed if present)
        - Normalized id_cols (deduplicated, with year_col and group_cols removed if present)
    """
    # 1. year_col 归一化
    y = (year_col or "").strip()
    y = y if y else None

    # 2. group_cols 归一化 (already deduplicated by _parse_csv_list)
    # Remove year_col from group_cols if present (同名去重)
    groups = [g for g in group_cols if g != y]

    # 3. id_cols 归一化 (already deduplicated by _parse_csv_list)
    # Remove year_col and group_cols from id_cols to avoid duplicates in final ID list
    seen = {y} if y else set()
    seen.update(groups)
    ids = [i for i in id_cols if i not in seen]

    # Verify all columns exist in dataframe
    all_input_cols = [c for c in [y] + groups + ids if c]
    missing_cols = [col for col in all_input_cols if col not in df_columns]
    if missing_cols:
        raise ValueError(f"以下列不存在于数据中：{missing_cols}")

    return y, groups, ids


def _determine_branch_logic(year_col: Optional[str], group_cols: List[str]) -> str:
    """
    Determine processing branch based on inputs.

    Returns:
        - "no_group_no_year": Neither group_cols nor year_col provided
        - "year_only": Only year_col provided
        - "group_only": Only group_cols provided
        - "group_year": Both group_cols and year_col provided
    """
    has_year = bool(year_col)
    has_groups = bool(group_cols)

    if not has_groups and not has_year:
        return "no_group_no_year"
    elif not has_groups and has_year:
        return "year_only"
    elif has_groups and not has_year:
        return "group_only"
    else:  # has_groups and has_year
        return "group_year"


def _get_final_id_columns(year_col: Optional[str], group_cols: List[str], id_cols: List[str]) -> List[str]:
    """
    Get final ID column order according to output_spec_entropy_topsis.md:
    1. year_col (if provided)
    2. group_cols (normalized order)
    3. Remaining id_cols
    """
    final_cols = []
    if year_col:
        final_cols.append(year_col)
    final_cols.extend(group_cols)

    # Add id_cols that are not already in the list
    used_cols = set(final_cols)
    for col in id_cols:
        if col not in used_cols:
            final_cols.append(col)
            used_cols.add(col)

    return final_cols


def _run_unified_topsis(
    df: pd.DataFrame,
    year_col: Optional[str],
    group_cols: List[str],
    id_cols: List[str],
    negative_indicators: Set[str],
    eps_shift: float,
    append_weights: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run TOPSIS calculation based on branch logic.

    Returns:
        - entropy_df: Final entropy table (with weight row if requested)
        - topsis_df: TOPSIS results table
        - weights_df: Weights table (may be empty for grouped runs)
    """
    branch = _determine_branch_logic(year_col, group_cols)
    final_id_cols = _get_final_id_columns(year_col, group_cols, id_cols)

    # Calculate all metric columns (exclude ID columns)
    id_col_set = set(final_id_cols)
    metric_cols = [col for col in df.columns if col not in id_col_set]

    if branch in ["no_group_no_year", "year_only"]:
        # No grouping - run on entire table
        if final_id_cols:
            result = run_entropy_topsis(
                df,
                id_cols=final_id_cols,
                negative_cols=negative_indicators,
                eps_shift=eps_shift
            )
        else:
            result = run_entropy_topsis(
                df,
                negative_cols=negative_indicators,
                eps_shift=eps_shift
            )

        # Append weights row if requested (for compatibility with legacy scripts)
        if append_weights and not result.weights.empty:
            weights_series = result.weights.iloc[0]
            entropy_df = append_weights_row(
                result.entropy,
                id_cols=final_id_cols[:1],  # Use first ID column for weight label
                weights=weights_series
            )
        else:
            entropy_df = result.entropy

        return entropy_df, result.topsis, result.weights

    else:  # "group_only" or "group_year"
        if final_id_cols:
            result = run_entropy_topsis(
                df,
                id_cols=final_id_cols,
                negative_cols=negative_indicators,
                eps_shift=eps_shift,
                group_cols=group_cols,
                year_col=year_col
            )
        else:
            result = run_entropy_topsis(
                df,
                negative_cols=negative_indicators,
                eps_shift=eps_shift,
                group_cols=group_cols,
                year_col=year_col
            )

        # Grouped runs should NOT append weights row according to spec
        return result.entropy, result.topsis, result.weights


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="""
统一熵权 TOPSIS 计算脚本

替代 4 个旧脚本的统一接口，支持多列分组和灵活的 ID 列配置。

分支逻辑：
- 无分组无年份：对全表计算一次
- 仅年份：对全表计算，年份作为 ID 列保留
- 仅分组：按分组列联合分组计算
- 分组+年份：按分组列联合分组，年份作为 ID 列保留
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 Excel 文件路径（默认: unified_topsis.xlsx）",
    )
    parser.add_argument(
        "--group-cols",
        default="",
        help="分组列名（逗号分隔，支持多列联合分组。例如：省份,行业）",
    )
    parser.add_argument(
        "--year-col",
        default=None,
        help="年份列名（仅作为 ID 列保留，不参与分组和指标计算）",
    )
    parser.add_argument(
        "--id-cols",
        default="",
        help="额外 ID 列名（逗号分隔，不参与指标计算但需保留。例如：学校名称）",
    )
    parser.add_argument(
        "--negative-indicators",
        default="",
        help="负向指标列名（逗号或换行分隔，留空使用默认列表）",
    )
    parser.add_argument(
        "--eps-shift",
        type=float,
        default=0.01,
        help="非负平移常数（默认: 0.01）",
    )
    parser.add_argument(
        "--append-weights",
        action="store_true",
        help="在熵权表末尾追加权重行（兼容单省份、年份脚本）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )

    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        # Setup run directory
        enter_run_dir(args.run_dir)

        # Prepare input/output paths
        input_file = require_input_path(
            args.input_file,
            "输入 Excel 文件",
            allowed_suffixes={".xlsx", ".xls"},
        )
        output_file = prepare_output_path(args.output_file, "unified_topsis.xlsx")

        # Load data
        df = pd.read_excel(input_file)
        df.columns = df.columns.map(lambda x: str(x).strip())

        # Parse inputs
        group_cols_raw = _parse_csv_list(args.group_cols)
        id_cols_raw = _parse_csv_list(args.id_cols)
        negative_indicators = parse_negative_indicators(args.negative_indicators)

        # Normalize inputs and handle duplicates
        year_col_norm, group_cols_norm, id_cols_norm = _normalize_inputs(
            args.year_col, group_cols_raw, id_cols_raw, df.columns.tolist()
        )

        # Show processing info
        branch = _determine_branch_logic(year_col_norm, group_cols_norm)
        final_id_cols = _get_final_id_columns(year_col_norm, group_cols_norm, id_cols_norm)

        print(f"处理模式: {branch}")
        print(f"年份列: {year_col_norm}")
        print(f"分组列: {group_cols_norm}")
        print(f"其他 ID 列: {id_cols_norm}")
        print(f"最终 ID 列顺序: {final_id_cols}")

        # Run TOPSIS calculation
        entropy_df, topsis_df, weights_df = _run_unified_topsis(
            df,
            year_col_norm,
            group_cols_norm,
            id_cols_norm,
            negative_indicators,
            args.eps_shift,
            args.append_weights
        )

        # Save results (legacy-compatible: only "熵权" and "topsis" sheets)
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            entropy_df.to_excel(writer, sheet_name="熵权", index=False)
            topsis_df.to_excel(writer, sheet_name="topsis", index=False)

        print(f"计算完成，结果已保存到: {output_file}")

    except Exception as exc:
        print(f"计算失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
