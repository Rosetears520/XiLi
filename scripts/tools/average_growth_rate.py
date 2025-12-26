from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


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


def _calculate_average_growth_rate(values: np.ndarray) -> float:
    # Legacy fallback: assume consecutive steps (n = count-1).
    clean = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(clean) < 2:
        return 0.0
    initial = float(clean[0])
    final = float(clean[-1])
    n = len(clean) - 1
    if n <= 0 or initial == 0:
        return 0.0
    if initial * final < 0:
        return 0.0
    return (final / initial) ** (1.0 / n) - 1.0


def _calculate_cagr(values: np.ndarray, years: np.ndarray) -> float:
    """Compute CAGR using actual year span rather than count of values.

    This fixes the bug where missing years (e.g. 2020 -> 2022) were treated as 1 step.
    """
    if len(values) != len(years):
        raise ValueError("values 与 years 长度不一致。")

    mask = ~np.isnan(values) & ~np.isnan(years)
    if mask.sum() < 2:
        return 0.0

    valid_values = values[mask].astype(float)
    valid_years = years[mask].astype(float)

    start_val = float(valid_values[0])
    end_val = float(valid_values[-1])
    start_year = float(valid_years[0])
    end_year = float(valid_years[-1])
    n = end_year - start_year

    if n <= 0 or start_val == 0:
        return 0.0
    if start_val * end_val < 0:
        return 0.0
    return (end_val / start_val) ** (1.0 / n) - 1.0


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
        help="需要填充的列（逗号分隔；为空则自动选择数值列且排除 id/group/sort 列）",
    )
    parser.add_argument(
        "--id-cols",
        default="",
        help="不参与填充的列（逗号分隔，例如: 年份,省份）",
    )
    parser.add_argument(
        "--round",
        action="store_true",
        help="对填充结果四舍五入（适合计数列）",
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
    group_cols: list[str],
    sort_col: Optional[str],
    cols: list[str],
    id_cols: list[str],
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
    if sort_col_clean is None:
        if "年份" in df.columns and "年份" not in group_cols:
            sort_col_clean = "年份"
        else:
            raise ValueError("必须提供 --sort-col（年份列），用于修正年均增长率计算。")

    id_cols_set = {c for c in id_cols if c}
    id_cols_set.update(group_cols)
    if sort_col_clean:
        id_cols_set.add(sort_col_clean)

    if cols:
        cols_to_fill = [c for c in cols if c]
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cols_to_fill = [c for c in numeric_cols if c not in id_cols_set]

    if not cols_to_fill:
        raise ValueError("未找到可用于填充的列，请通过 --cols 指定需要填充的列。")

    for col in cols_to_fill:
        if col not in df.columns:
            raise ValueError(f"填充列不存在：{col}")

    work = df.copy()
    if sort_col_clean:
        work = work.sort_values(by=[*group_cols, sort_col_clean], kind="mergesort")

    def _fill_group(group: pd.DataFrame) -> pd.DataFrame:
        years = pd.to_numeric(group[sort_col_clean], errors="coerce").to_numpy(dtype=float)
        for col in cols_to_fill:
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            if not np.isnan(values).any():
                continue

            growth_rate = _calculate_cagr(values, years)

            valid_indices = np.where(~np.isnan(values))[0]
            if len(valid_indices) == 0:
                continue
            first_valid_idx = int(valid_indices[0])

            # Forward fill: Current = Previous * (1 + r)^(delta_year)
            for i in range(first_valid_idx + 1, len(values)):
                if np.isnan(values[i]):
                    prev = float(values[i - 1])
                    if np.isnan(prev):
                        continue
                    delta = years[i] - years[i - 1]
                    if not np.isfinite(delta) or delta <= 0:
                        delta = 1.0
                    values[i] = prev * (1.0 + growth_rate) ** float(delta)

            # Backward fill: Previous = Current / (1 + r)^(delta_year)
            for i in range(first_valid_idx - 1, -1, -1):
                if np.isnan(values[i]):
                    nxt = float(values[i + 1])
                    if np.isnan(nxt):
                        continue
                    delta = years[i + 1] - years[i]
                    if not np.isfinite(delta) or delta <= 0:
                        delta = 1.0
                    values[i] = nxt / (1.0 + growth_rate) ** float(delta)

            if round_values:
                values = np.round(values)
            group[col] = values
        return group

    result = work.groupby(group_cols, dropna=False, sort=False).apply(_fill_group)
    result = result.reset_index(drop=True)

    out_suffix = output_file.suffix.lower()
    if out_suffix == ".csv":
        result.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        result.to_excel(output_file, index=False)
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
        main(
            input_file,
            output_file,
            group_cols=_parse_csv_list(args.group_cols),
            sort_col=args.sort_col,
            cols=_parse_csv_list(args.cols),
            id_cols=_parse_csv_list(args.id_cols),
            round_values=bool(args.round),
        )
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
