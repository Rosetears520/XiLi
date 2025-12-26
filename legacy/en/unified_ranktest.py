#!/usr/bin/env python3
"""
Unified Rank Test CLI Script

This script provides a unified interface for rank sum tests (Mann-Whitney U, Kruskal-Wallis H),
replacing the 2 legacy scripts with a single comprehensive command-line interface.

Supported legacy script equivalents:
1. 秩和检验.py -> --mode questions --question-cols Q1,Q2,Q3 --dv-col BIU
2. 秩和检验---分组.py -> --mode group --group-cols 区域

Mode logic:
- questions: Perform rank sum tests for each question column against the dependent variable
- group: Perform rank sum tests for grouping comparisons (single or multiple columns)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path
from xili.rank_tests import run_rank_tests


def _parse_csv_list(text: Optional[str]) -> list[str]:
    """Parse comma-separated list into list of strings, handling whitespace."""
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="""
统一秩和检验计算脚本

替代 2 个旧脚本的统一接口，支持单组和多组秩和检验。

模式逻辑：
- questions：对每个问题列与因变量进行秩和检验
- group：对分组列进行组间秩和检验（支持多列联合分组）
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
        help="输出 Excel 文件路径（默认: unified_ranktest.xlsx）",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["questions", "group"],
        help="检验模式（必填）：questions 或 group",
    )
    parser.add_argument(
        "--question-cols",
        default="",
        help="questions 模式必填：自变量列名（逗号分隔，例如: Q1,Q2,Q3）",
    )
    parser.add_argument(
        "--dv-col",
        default="BIU",
        help="questions 模式可选：因变量列名（默认: BIU，与旧脚本一致）",
    )
    parser.add_argument(
        "--group-cols",
        default="",
        help="group 模式可选：分组列名（逗号分隔，支持多列；默认: 区域，与旧脚本一致）",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="兼容性参数：单个分组列名（如果同时提供了 --group-cols，则以 --group-cols 为准）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="显著性水平 alpha（默认: 0.05）",
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
        output_file = prepare_output_path(args.output_file, "unified_ranktest.xlsx")

        # Load data
        df = pd.read_excel(input_file)

        mode = (args.mode or "").strip().lower()
        if mode == "questions":
            question_cols = _parse_csv_list(args.question_cols)
            if not question_cols:
                raise ValueError("mode=questions 时必须指定 --question-cols，例如 --question-cols Q1,Q2,Q3")
            result_df = run_rank_tests(
                df,
                mode="questions",
                question_cols=question_cols,
                dv_col=args.dv_col,
                alpha=float(args.alpha),
            )
        elif mode == "group":
            if (args.question_cols or "").strip():
                raise ValueError("mode=group 时不应提供 --question-cols")

            # Handle group_cols vs group-col with backward compatibility
            group_cols = _parse_csv_list(args.group_cols)
            if group_cols:
                # Use group_cols if provided
                if args.group_col and args.group_col.strip():
                    print("注意：同时提供了 --group-cols 和 --group-col，将使用 --group-cols", file=sys.stderr)
                group_cols_param = group_cols
                group_col_param = None
            else:
                # Fall back to single group_col for backward compatibility
                group_col_param = args.group_col or "区域"
                group_cols_param = None

            result_df = run_rank_tests(
                df,
                mode="group",
                group_col=group_col_param,
                group_cols=group_cols_param,
                alpha=float(args.alpha),
            )
        else:
            raise ValueError(f"未知 mode：{args.mode}")

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            result_df.to_excel(writer, sheet_name="Summary", index=False)

        print(f"分析完成，结果已写入 {output_file}（sheet: Summary）。")
        print(result_df)

    except Exception as exc:
        print(f"计算失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())