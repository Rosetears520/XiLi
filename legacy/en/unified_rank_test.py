#!/usr/bin/env python3
"""
统一秩和检验 CLI 脚本

此脚本合并了以下两个旧脚本的功能：
- 秩和检验.py (questions 模式)
- 秩和检验---分组.py (group 模式)

支持模式：
1. questions 模式：对每个问题列，检验不同组的因变量是否有显著差异
2. group 模式：对除分组列外的所有变量，检验不同组别间是否有显著差异

使用示例：
# questions 模式 (原 秩和检验.py)
python unified_rank_test.py questions --input-file data.xlsx --question-cols Q1,Q2,Q3 --dv-col BIU

# group 模式 - 单分组列 (原 秩和检验---分组.py)
python unified_rank_test.py group --input-file data.xlsx --group-col 区域

# group 模式 - 多分组列 (新功能)
python unified_rank_test.py group --input-file data.xlsx --group-cols 区域,年份

# 直接使用 python -m 方式
python -m xili.cli ranktest --mode questions --input-file data.xlsx --question-cols Q1,Q2,Q3
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path
from xili.rank_tests import run_rank_tests


def _parse_csv_list(text: Optional[str]) -> list[str]:
    """解析逗号分隔的列名列表"""
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="统一秩和检验工具（支持 questions 和 group 两种模式）",
        epilog="""
示例用法：
  questions 模式：
    python unified_rank_test.py questions --input-file data.xlsx --question-cols Q1,Q2,Q3 --dv-col BIU

  group 模式（单列）：
    python unified_rank_test.py group --input-file data.xlsx --group-col 区域

  group 模式（多列）：
    python unified_rank_test.py group --input-file data.xlsx --group-cols 区域,年份
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional argument for mode
    parser.add_argument(
        "mode",
        choices=["questions", "group"],
        help="检验模式（必填）: questions 或 group",
    )

    # Common arguments
    parser.add_argument(
        "--input-file",
        required=True,
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 Excel 文件路径（默认: rank_test_summary.xlsx for questions, group_rank_test_results.xlsx for group）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="显著性水平（默认: 0.05）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )

    # Questions mode arguments
    questions_group = parser.add_argument_group("questions 模式参数")
    questions_group.add_argument(
        "--question-cols",
        default="",
        help="questions 模式必填：自变量列名（逗号分隔，例如: Q1,Q2,Q3）",
    )
    questions_group.add_argument(
        "--dv-col",
        default="BIU",
        help="questions 模式可选：因变量列名（默认: BIU）",
    )

    # Group mode arguments
    group_group = parser.add_argument_group("group 模式参数")
    group_group.add_argument(
        "--group-cols",
        default="",
        help="group 模式可选：分组列名（逗号分隔，支持多列；默认: 区域）",
    )
    group_group.add_argument(
        "--group-col",
        default=None,
        help="兼容性参数：单个分组列名（如果同时提供了 --group-cols，则以 --group-cols 为准）",
    )

    return parser


def main() -> None:
    """主函数"""
    parser = build_parser()
    args = parser.parse_args()

    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入 Excel 文件",
            allowed_suffixes={".xlsx", ".xls"},
        )

        # Set default output filename based on mode
        if args.output_file is None:
            if args.mode == "questions":
                args.output_file = "rank_test_summary.xlsx"
            else:  # group
                args.output_file = "group_rank_test_results.xlsx"

        output_file = prepare_output_path(args.output_file, args.output_file)
        df = pd.read_excel(input_file)

        if args.mode == "questions":
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

            # Save results
            result_df.to_excel(output_file, sheet_name="Summary", index=False)

            print(f"questions 模式分析完成，结果已写入 {output_file}")
            print("【Summary 表】示例预览：")
            print(result_df)

        elif args.mode == "group":
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

            # Save results
            result_df.to_excel(output_file, index=False)

            print(f"group 模式分析完成，结果已写入 {output_file}")
            print("【检验结果】预览：")
            print(result_df)

    except Exception as exc:
        print(f"分析失败: {exc}", file=sys.stderr)
        sys.exit(1)


def cli(argv: Optional[list[str]] = None) -> int:
    """CLI 接口"""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入 Excel 文件",
            allowed_suffixes={".xlsx", ".xls"},
        )

        # Set default output filename based on mode
        if args.output_file is None:
            if args.mode == "questions":
                args.output_file = "rank_test_summary.xlsx"
            else:  # group
                args.output_file = "group_rank_test_results.xlsx"

        output_file = prepare_output_path(args.output_file, args.output_file)
        df = pd.read_excel(input_file)

        if args.mode == "questions":
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

            # Save results
            result_df.to_excel(output_file, sheet_name="Summary", index=False)

            print(f"questions 模式分析完成，结果已写入 {output_file}")
            print("【Summary 表】示例预览：")
            print(result_df)

        elif args.mode == "group":
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

            # Save results
            result_df.to_excel(output_file, index=False)

            print(f"group 模式分析完成，结果已写入 {output_file}")
            print("【检验结果】预览：")
            print(result_df)

    except Exception as exc:
        print(f"分析失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    main()