from __future__ import annotations

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
        help="输入 CSV/Excel 路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 CSV/Excel 路径（默认: linear_interpolation_result.csv）",
    )
    parser.add_argument(
        "--group-cols",
        default="",
        help="分组列（逗号分隔，例如: 省份）",
    )
    parser.add_argument(
        "--sort-col",
        default=None,
        help="组内排序列（可选，例如: 年份）",
    )
    parser.add_argument(
        "--cols",
        default="",
        help="需要插值的列（逗号分隔；为空则自动选择数值列且排除 id/group/sort 列）",
    )
    parser.add_argument(
        "--id-cols",
        default="",
        help="不进行插值的列（逗号分隔，例如: 年份,省份）",
    )
    parser.add_argument(
        "--limit-direction",
        default="forward",
        choices=["forward", "both"],
        help="插值方向（默认: forward；both 会同时填充首尾缺失）",
    )
    parser.add_argument(
        "--limit-area",
        default="inside",
        choices=["inside", "none"],
        help="插值范围（默认: inside 仅填补中间缺口；none 不限制）",
    )
    parser.add_argument(
        "--round",
        action="store_true",
        help="对插值结果四舍五入（适合计数列）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def _read_table(input_file: Path) -> pd.DataFrame:
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        for encoding in ["utf-8", "gbk", "ansi", "utf-8-sig"]:
            try:
                return pd.read_csv(input_file, encoding=encoding)
            except Exception:
                continue
        return pd.read_csv(input_file)
    return pd.read_excel(input_file)


def main(
    input_file: Path,
    output_file: Path,
    *,
    group_cols: list[str],
    sort_col: Optional[str],
    cols: list[str],
    id_cols: list[str],
    limit_direction: str,
    limit_area: str,
    round_values: bool,
) -> None:
    df = _read_table(input_file)
    df.columns = df.columns.map(lambda x: str(x).strip())

    group_cols = [c for c in group_cols if c]
    if not group_cols:
        raise ValueError("必须指定 --group-cols，例如 --group-cols 省份")
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"分组列不存在：{col}")

    sort_col_clean = (sort_col or "").strip()
    sort_col_clean = sort_col_clean if sort_col_clean else None
    if sort_col_clean and sort_col_clean not in df.columns:
        raise ValueError(f"排序列不存在：{sort_col_clean}")

    id_cols_set = {c for c in id_cols if c}
    id_cols_set.update(group_cols)
    if sort_col_clean:
        id_cols_set.add(sort_col_clean)

    if cols:
        cols_to_interpolate = [c for c in cols if c]
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cols_to_interpolate = [c for c in numeric_cols if c not in id_cols_set]

    if not cols_to_interpolate:
        raise ValueError("未找到可用于插值的列，请通过 --cols 指定需要插值的列。")

    for col in cols_to_interpolate:
        if col not in df.columns:
            raise ValueError(f"插值列不存在：{col}")

    if limit_direction not in {"forward", "both"}:
        raise ValueError(f"未知 limit_direction：{limit_direction}")
    if limit_area not in {"inside", "none"}:
        raise ValueError(f"未知 limit_area：{limit_area}")

    work = df.copy()
    if sort_col_clean:
        work = work.sort_values(by=[*group_cols, sort_col_clean], kind="mergesort")

    def _interpolate_group(group: pd.DataFrame) -> pd.DataFrame:
        for col in cols_to_interpolate:
            series = pd.to_numeric(group[col], errors="coerce")
            group[col] = series.interpolate(
                method="linear",
                limit_direction=limit_direction,
                limit_area=None if limit_area == "none" else limit_area,
            )
            if round_values:
                group[col] = group[col].round()
        return group

    result = work.groupby(group_cols, dropna=False, sort=False).apply(_interpolate_group)
    result = result.reset_index(drop=True)

    out_suffix = output_file.suffix.lower()
    if out_suffix == ".csv":
        result.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        result.to_excel(output_file, index=False)


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入 CSV/Excel 文件",
            allowed_suffixes={".csv", ".xlsx", ".xls"},
        )
        output_file = prepare_output_path(args.output_file, "linear_interpolation_result.csv")
        main(
            input_file,
            output_file,
            group_cols=_parse_csv_list(args.group_cols),
            sort_col=args.sort_col,
            cols=_parse_csv_list(args.cols),
            id_cols=_parse_csv_list(args.id_cols),
            limit_direction=str(args.limit_direction),
            limit_area=str(args.limit_area),
            round_values=bool(args.round),
        )
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
