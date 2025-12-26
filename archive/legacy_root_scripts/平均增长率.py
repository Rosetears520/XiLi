import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path

def calculate_average_growth_rate(values):
    """计算年均增长率"""
    initial_value = values[0]
    final_value = values[-1]
    n = len(values) - 1
    average_growth_rate = (final_value / initial_value) ** (1 / n) - 1
    return average_growth_rate

def fill_missing_values_by_growth_rate(df):
    """按省份分组，检测空缺值并使用年均增长率填充"""
    # 按省份分组
    grouped = df.groupby('省份')

    for province, group in grouped:
        # 获取省份的数据列（假设U1, U2, U3, M1列是数值列）
        province_values = group[[
            'M1', 'S2', 'S3', 'A2']].values

        # 对每列数据进行检测
        for col_index, col_name in enumerate(
                ['M1', 'S2', 'S3', 'A2']):
            values = province_values[:, col_index]

            # 检测是否有空缺值
            if np.any(pd.isnull(values)):
                # 计算年均增长率
                growth_rate = calculate_average_growth_rate(values[~pd.isnull(values)])  # 排除空值
                # 使用年均增长率填充空缺值
                for i in range(len(values)):
                    if pd.isnull(values[i]):
                        # 使用年均增长率填充
                        if i == 0:
                            values[i] = values[~pd.isnull(values)].mean()  # 用第一个有效值填充
                        else:
                            values[i] = values[i - 1] * (1 + growth_rate)

                # 更新数据框中的值
                df.loc[group.index, col_name] = values

    return df

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="平均增长率填充工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 csv 和 excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 csv 和 excel 文件路径（默认: average_growth_rate_result.csv）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(input_file: Path, output_file: Path) -> None:
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        # Try different encodings for CSV
        df = None
        for encoding in ["utf-8", "gbk", "ansi", "utf-8-sig"]:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                break
            except Exception:
                continue
        if df is None:
            df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)

    df_filled = fill_missing_values_by_growth_rate(df)
    
    out_suffix = output_file.suffix.lower()
    if out_suffix == ".csv":
        df_filled.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        df_filled.to_excel(output_file, index=False)
    
    print(f"处理后的数据已保存到: {output_file}")


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入 csv 和 excel 文件",
            allowed_suffixes={".csv", ".xlsx", ".xls"},
        )
        output_file = prepare_output_path(args.output_file, "average_growth_rate_result.csv")
        main(input_file, output_file)
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
