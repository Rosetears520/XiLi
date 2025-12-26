from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def _read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        for encoding in ["utf-8", "gbk", "ansi", "utf-8-sig"]:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except Exception:
                continue
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def _parse_cols(value: str) -> list[str]:
    return [c.strip() for c in (value or "").split(",") if c.strip()]


def _parse_rename_pairs(pairs: Optional[list[str]], cols: list[str]) -> dict[str, str]:
    if not pairs:
        return {col: f"{col}_avg" for col in cols}
    mapping: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"重命名参数格式错误: {item}，应为 old=new")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="计算分组平均值工具（默认优先按 年份 分组）")
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
        "--group-col",
        default=None,
        help="分组列（可选；默认：若存在 年份 则用 年份，否则若存在 股票代码 则用 股票代码，否则不分组）",
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


def main(
    input_file: Path,
    output_file: Path,
    *,
    group_col: Optional[str],
    columns_to_average: list[str],
    new_column_names: dict[str, str],
) -> None:
    df = _read_table(input_file)
    df.columns = df.columns.map(lambda x: str(x).strip())

    if not columns_to_average:
        raise ValueError("必须指定 --cols 列表，例如 --cols T,O,E")

    for col in columns_to_average:
        if col not in df.columns:
            raise ValueError(f"平均值列不存在：{col}")

    group_col_clean = (group_col or "").strip()
    if not group_col_clean:
        if "年份" in df.columns:
            group_col_clean = "年份"
        elif "股票代码" in df.columns:
            group_col_clean = "股票代码"
        else:
            group_col_clean = ""

    numeric_df = df[columns_to_average].apply(pd.to_numeric, errors="coerce")
    base = df.copy()
    for col in columns_to_average:
        base[col] = numeric_df[col]

    if group_col_clean:
        if group_col_clean not in df.columns:
            raise ValueError(f"分组列不存在：{group_col_clean}")
        result = base.groupby(group_col_clean)[columns_to_average].mean()
    else:
        result = pd.DataFrame([numeric_df.mean().to_dict()])

    result = result.rename(columns=new_column_names)

    out_suffix = output_file.suffix.lower()
    if out_suffix == ".csv":
        result.to_csv(output_file, index=bool(group_col_clean), encoding="utf-8-sig")
    else:
        result.to_excel(output_file, index=bool(group_col_clean))
    print(f"处理完成，结果已保存至 {output_file}")


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file_path = require_input_path(
            args.input_file,
            "输入 csv 和 excel 文件",
            allowed_suffixes={".xlsx", ".xls", ".csv"},
        )
        output_file_path = prepare_output_path(args.output_file, "yearly_averages.xlsx")
        cols_to_avg = _parse_cols(args.cols)
        new_cols_names = _parse_rename_pairs(args.rename, cols_to_avg)

        main(
            input_file=input_file_path,
            output_file=output_file_path,
            group_col=args.group_col,
            columns_to_average=cols_to_avg,
            new_column_names=new_cols_names,
        )
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
