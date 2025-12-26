import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="熵权 TOPSIS（单省份/单主体）")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 Excel 文件路径（默认: topsis_single.xlsx）",
    )
    parser.add_argument(
        "--name-col",
        default="学校名称",
        help="名称列名（默认: 学校名称）",
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
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(
    input_file_path: Path,
    output_file_path: Path,
    name_col: str,
    eps_shift: float,
    negative_indicators_text: Optional[str],
) -> None:
    # Get absolute path to unified script
    script_dir = Path(__file__).parent
    unified_script = script_dir / "unified_topsis.py"

    # Build command for unified TOPSIS script
    cmd = [
        sys.executable, str(unified_script),
        "--input-file", str(input_file_path),
        "--output-file", str(output_file_path),
        "--id-cols", name_col or "学校名称",
        "--eps-shift", str(eps_shift),
        "--append-weights",  # Enable weight append for compatibility with original script
    ]

    if negative_indicators_text:
        cmd.extend(["--negative-indicators", negative_indicators_text])

    # Run the unified TOPSIS script
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"统一TOPSIS脚本执行失败:\n{result.stderr}")

    # Print the same message as original
    print("计算完成，最终结果已保存到文件：", output_file_path)


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_path = require_input_path(
            args.input_file,
            "输入 Excel 文件",
            allowed_suffixes={".xlsx", ".xls"},
        )
        output_path = prepare_output_path(args.output_file, "topsis_single.xlsx")
        main(
            input_path,
            output_path,
            args.name_col,
            args.eps_shift,
            args.negative_indicators,
        )
    except Exception as exc:
        print(f"计算失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
