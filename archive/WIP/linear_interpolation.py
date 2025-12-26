import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="线性插值处理工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 CSV 路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 CSV 路径（默认: linear_interpolation_result.csv）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(input_file: Path, output_file: Path) -> None:
    df = pd.read_csv(input_file, encoding="ANSI")

    # 确保年份列为整数类型
    df["年份"] = df["年份"].astype(int)

    # 需要进行插值的列名列表
    columns_to_interpolate = [
        "高新技术企业数量 (U1)",
    ]

    # 按省份分组，并对每个需要插值的列应用线性插值
    for column in columns_to_interpolate:
        df[column] = df.groupby("省份")[column].transform(lambda group: group.interpolate(method="linear"))

    # 四舍五入'高新技术企业数量 (U1)'列的插值结果
    df["高新技术企业数量 (U1)"] = df["高新技术企业数量 (U1)"].round()

    df.to_csv(output_file, index=False, encoding="ANSI")


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入 CSV 文件",
            allowed_suffixes={".csv"},
        )
        output_file = prepare_output_path(args.output_file, "linear_interpolation_result.csv")
        main(input_file, output_file)
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
