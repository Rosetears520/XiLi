import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def read_data(file_path):
    """
    从指定的 csv 或 excel 文件路径读取数据。

    Args:
        file_path (str): 输入文件的路径。

    Returns:
        pandas.DataFrame: 从文件读取到的数据。如果文件未找到或出错，则返回 None。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            for encoding in ["utf-8", "gbk", "utf-8-sig"]:
                try:
                    return pd.read_csv(path, encoding=encoding)
                except Exception:
                    continue
            return pd.read_csv(path)
        return pd.read_excel(path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def preprocess_data(df):
    """
    通过清理指定列来预处理数据。

    Args:
        df (pandas.DataFrame): 输入的DataFrame。

    Returns:
        pandas.DataFrame: 经过预处理的DataFrame。
    """
    # 这个函数处理数据清洗工作。
    # 在这里，我们移除了'股票代码'列中的方括号。
    if '股票代码' in df.columns:
        # 使用 .str.replace 方法替换字符串，regex=False表示不使用正则表达式
        df['股票代码'] = df['股票代码'].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    return df


def calculate_averages(df, columns_to_average):
    """
    为每个股票代码计算指定列的平均值。

    Args:
        df (pandas.DataFrame): 用于执行计算的DataFrame。
        columns_to_average (list): 需要计算平均值的列名列表。

    Returns:
        pandas.DataFrame: 一个包含计算出的平均值的新DataFrame。
    """
    # 这个函数执行主要的计算任务。
    # 它按'股票代码'对数据进行分组，然后计算平均值。
    if '股票代码' in df.columns:
        average_values = df.groupby('股票代码')[columns_to_average].mean()
        return average_values
    else:
        print("错误：'股票代码' 列不存在。")
        return None


def rename_columns(df, new_column_names):
    """
    重命名DataFrame的列。

    Args:
        df (pandas.DataFrame): 需要重命名列的DataFrame。
        new_column_names (dict): 一个将旧列名映射到新列名的字典。

    Returns:
        pandas.DataFrame: 列被重命名后的DataFrame。
    """
    # 为最终输出重命名列。
    return df.rename(columns=new_column_names)


def save_data(df, output_path):
    """
    将 DataFrame 保存到 csv 或 excel 文件。

    Args:
        df (pandas.DataFrame): 需要保存的DataFrame。
        output_path (str): 输出文件的路径。
    """
    path = Path(output_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df.to_csv(path, index=True, encoding="utf-8-sig")
        else:
            df.to_excel(path)
        print(f"处理完成，结果已保存至 {output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


def main(
    input_file: Path,
    output_file: Path,
    columns_to_average: list[str],
    new_column_names: dict[str, str],
) -> None:
    """
    主函数，用于协调整个数据处理流程。
    现在它接收配置作为参数。
    """
    # --- 数据处理流水线 ---
    # 1. 读取数据
    df = read_data(str(input_file))

    if df is not None:
        # 2. 预处理数据
        df = preprocess_data(df)

        # --- 扩展点 ---
        # 你可以在这里添加你自己的数据处理函数。
        # 例如:
        # df = your_custom_function(df)
        # -------------------------

        # 3. 计算平均值
        avg_df = calculate_averages(df, columns_to_average)

        if avg_df is not None:
            # 4. 重命名列
            avg_df = rename_columns(avg_df, new_column_names)

            # 5. 保存结果
            save_data(avg_df, str(output_file))

            print("\n计算出的平均值为:")
            print(avg_df)


# 当该脚本被直接执行时，`__name__` 的值是 `__main__`
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="计算年份平均值工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 csv 和 excel 文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 csv 和 excel 文件路径（默认: yearly_averages.xlsx）",
    )
    parser.add_argument(
        "--cols",
        default="",
        help="需要计算平均值的列（逗号分隔，例如: T,O,E）",
    )
    parser.add_argument(
        "--rename",
        action="append",
        default=None,
        help="列重命名，格式 old=new，可重复传入",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def _parse_cols(value: str) -> list[str]:
    return [c.strip() for c in (value or "").split(",") if c.strip()]


def _parse_rename_pairs(pairs: Optional[list[str]], cols: list[str]) -> dict:
    if not pairs:
        return {col: f"{col}_avg" for col in cols}
    mapping: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"重命名参数格式错误: {item}，应为 old=new")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file_path = require_input_path(
            args.input_file,
            "输入 csv 和 excel 文件",
            allowed_suffixes={ ".xlsx", ".xls", ".csv" },
        )
        output_file_path = prepare_output_path(args.output_file, "yearly_averages.xlsx")
        cols_to_avg = _parse_cols(args.cols)
        if not cols_to_avg:
            raise ValueError("必须指定 --cols 列表，例如 --cols T,O,E")
        new_cols_names = _parse_rename_pairs(args.rename, cols_to_avg)

        main(
            input_file=input_file_path,
            output_file=output_file_path,
            columns_to_average=cols_to_avg,
            new_column_names=new_cols_names,
        )
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
