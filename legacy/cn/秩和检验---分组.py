import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分组秩和检验工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 Excel 文件路径（默认: group_rank_test_results.xlsx）",
    )
    parser.add_argument(
        "--group-col",
        default="区域",
        help="分组列名（默认: 区域）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(input_file: Path, output_file: Path, group_col: str) -> None:
    # Build command for unified CLI
    # Get absolute path to unified script
    script_dir = Path(__file__).parent
    unified_script = script_dir / "unified_ranktest.py"

    # Build command for unified rank test script
    cmd = [
        sys.executable, str(unified_script),
        "--mode", "group",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--group-cols", group_col,
    ]

    # Run the unified CLI
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"统一秩和检验脚本执行失败:\n{result.stderr}")

    # Print the same messages as original
    print("检验结果：")

    # Display the results if the unified CLI printed them
    if result.stdout:
        print(result.stdout)


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
        output_file = prepare_output_path(args.output_file, "group_rank_test_results.xlsx")
        main(input_file, output_file, args.group_col)
    except Exception as exc:
        print(f"分析失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
