import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="秩和检验工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 Excel 文件路径（默认: rank_test_summary.xlsx）",
    )
    parser.add_argument(
        "--question-cols",
        default="",
        help="自变量列名（逗号分隔，例如: Q1,Q2,Q3）",
    )
    parser.add_argument(
        "--dv-col",
        default="BIU",
        help="因变量列名（默认: BIU）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(input_file: Path, output_file: Path, question_cols: list[str], dv_col: str) -> None:
    # Get absolute path to unified script
    script_dir = Path(__file__).parent
    unified_script = script_dir / "unified_ranktest.py"

    # Build command for unified rank test script
    question_cols_str = ",".join(question_cols)

    cmd = [
        sys.executable, str(unified_script),
        "--mode", "questions",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--question-cols", question_cols_str,
        "--dv-col", dv_col,
    ]

    # Run the unified CLI
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"统一秩和检验脚本执行失败:\n{result.stderr}")

    # Print the same messages as original
    print(f"分析完成，结果已写入 {output_file}。")
    print("【Summary 表】示例预览：")

    # Display the summary if the unified CLI printed it
    if result.stdout:
        # Extract the summary table output from the CLI stdout
        lines = result.stdout.strip().split('\n')
        if lines:
            print('\n'.join(lines))


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入 Excel 文件",
            allowed_suffixes={".xlsx", ".xls"},
        )
        output_file = prepare_output_path(args.output_file, "rank_test_summary.xlsx")
        question_cols = [c.strip() for c in (args.question_cols or "").split(",") if c.strip()]
        if not question_cols:
            raise ValueError("必须指定 --question-cols，例如 --question-cols Q1,Q2,Q3")
        main(input_file, output_file, question_cols, args.dv_col)
    except Exception as exc:
        print(f"分析失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
