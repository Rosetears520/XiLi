import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="熵权 TOPSIS（按省份分组-年份）")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 Excel 文件路径（默认: topsis_by_province_year.xlsx）",
    )
    parser.add_argument(
        "--group-col",
        default="省份",
        help="分组列名（默认: 省份）",
    )
    parser.add_argument(
        "--year-col",
        default="年份",
        help="年份列名（默认: 年份）",
    )
    parser.add_argument(
        "--negative-indicators",
        default="",
        help="负向指标列名（逗号或换行分隔，留空使用默认列表）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(
    input_file: Path,
    output_file: Path,
    group_col: str,
    year_col: str,
    negative_indicators_text: Optional[str],
) -> None:
    group_col = (group_col or "省份").strip()
    year_col = (year_col or "年份").strip()

    # Get absolute path to unified script
    script_dir = Path(__file__).parent
    unified_script = script_dir / "unified_topsis.py"

    # Build command for unified TOPSIS script with grouping mode
    # Note: unified_topsis.py uses --group-cols (plural) not --group-col (singular)
    cmd = [
        sys.executable, str(unified_script),
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--group-cols", group_col,
        "--year-col", year_col,
    ]

    if negative_indicators_text:
        cmd.extend(["--negative-indicators", negative_indicators_text])

    # Run the unified TOPSIS script
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"统一TOPSIS脚本执行失败:\n{result.stderr}")

    # Print the same message as original
    print("计算完成，最终结果已保存到文件。")


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
        output_file = prepare_output_path(args.output_file, "topsis_by_province_year.xlsx")
        main(
            input_file,
            output_file,
            args.group_col,
            args.year_col,
            args.negative_indicators,
        )
    except Exception as exc:
        print(f"计算失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
