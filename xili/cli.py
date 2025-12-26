from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from common.cli_utils import prepare_output_path, require_input_path, run_command_capture
from common.run_context import run_context
from common.run_summary import write_run_summary
from xili.entropy_topsis import append_weights_row, run_entropy_topsis
from xili.rank_tests import run_rank_tests


def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from either CSV or Excel file."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        # Try different encodings for CSV
        for encoding in ["utf-8", "gbk", "utf-8-sig"]:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except Exception:
                continue
        return pd.read_csv(file_path)  # Fallback to default
    return pd.read_excel(file_path)


def _parse_csv_list(text: Optional[str]) -> list[str]:
    """解析逗号或换行分隔的列表，去除空格、过滤空项并保持顺序去重。"""
    if not text:
        return []
    import re

    parts = re.split(r"[,\n]", text)
    seen = set()
    result = []
    for p in parts:
        s = p.strip()
        if s and s not in seen:
            result.append(s)
            seen.add(s)
    return result


def _normalize_group_year_cols(
    year_col: Optional[str],
    group_cols_str: str,
    group_col_fallback: Optional[str] = None,
) -> tuple[Optional[str], list[str]]:
    """归一化 year_col 和 group_cols，处理同名去重与兼容性。"""
    # 1. year_col 归一化
    y = (year_col or "").strip()
    y = y if y else None

    # 2. group_cols 归一化
    groups = _parse_csv_list(group_cols_str)

    # 兼容性：如果未提供 group_cols 但提供了 group_col_fallback
    if not groups and group_col_fallback:
        f = (group_col_fallback or "").strip()
        if f:
            groups = [f]

    # 3. 同名去重：如果 year_col 在 group_cols 中，则从 group_cols 移除
    if y and y in groups:
        groups = [g for g in groups if g != y]

    return y, groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xili", description="统一数据分析 CLI（熵权TOPSIS + 秩和检验）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ===== topsis =====
    topsis_parser = subparsers.add_parser(
        "topsis",
        help="熵权 TOPSIS（可选分组）",
        description=(
            "熵权 TOPSIS（可选分组）。\n"
            "- 仅分组：提供 --group-cols/--group-col\n"
            "- 仅年份：提供 --year-col\n"
            "- 分组+年份：同时提供 --group-cols/--group-col 与 --year-col\n"
            "- 同名去重：group_cols 与 year_col 同名时自动去重"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    topsis_parser.add_argument("--input-file", default=None, help="输入 csv 和 excel 文件路径")
    topsis_parser.add_argument("--output-file", default=None, help="输出 Excel 文件路径（默认: entropy_topsis.xlsx）")
    topsis_parser.add_argument("--group-cols", default=None, help="分组列（逗号分隔，例如: 省份,行业）")
    topsis_parser.add_argument("--group-col", default=None, help="分组列（单列，兼容旧版）")
    topsis_parser.add_argument("--year-col", default=None, help="年份列")
    topsis_parser.add_argument("--negative", default=None, help="负指标列（逗号分隔）")
    topsis_parser.add_argument("--negative-indicators", default=None, help="负指标列（逗号分隔，同 --negative）")
    topsis_parser.add_argument("--id-cols", default=None, help="额外 ID 列（逗号分隔）")
    topsis_parser.add_argument("--eps-shift", type=float, default=0.01, help="熵权计算时的平移量（默认: 0.01）")
    topsis_parser.add_argument("--run-dir", default=None, help="运行输出目录（默认: runs/YYYYMMDD-HHMMSS_hash）")
    topsis_parser.add_argument("--append-weights", action="store_true", help="在熵权表末尾追加权重行（兼容旧脚本）")

    # ===== ranktest =====
    rank_parser = subparsers.add_parser(
        "ranktest",
        help="秩和检验（两组/多组）",
        description=(
            "秩和检验：\n"
            "- 两组：Mann-Whitney U（非参数、独立样本秩和检验）\n"
            "- 多组：Kruskal-Wallis H（非参数，一元ANOVA替代）\n\n"
            "模式（二选一，互斥）：\n"
            "- questions：--mode questions --question-cols Q1,Q2,... [--dv-col BIU]\n"
            "- group：--mode group [--group-col 区域]"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    rank_parser.add_argument("--input-file", default=None, help="输入 csv 和 excel 文件路径")
    rank_parser.add_argument("--output-file", default=None, help="输出 Excel 文件路径（默认: rank_tests.xlsx）")
    rank_parser.add_argument("--mode", required=True, choices=["questions", "group"], help="运行模式")
    rank_parser.add_argument("--question-cols", default=None, help="问题列（逗号分隔，questions 模式必填）")
    rank_parser.add_argument("--dv-col", default=None, help="因变量列（questions 模式可选）")
    rank_parser.add_argument("--group-cols", default=None, help="分组列（逗号分隔，group 模式）")
    rank_parser.add_argument("--group-col", default=None, help="分组列（单列，兼容旧版）")
    rank_parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平（默认: 0.05）")
    rank_parser.add_argument("--run-dir", default=None, help="运行输出目录（默认: runs/YYYYMMDD-HHMMSS_hash）")

    # ===== selfcheck =====
    selfcheck_parser = subparsers.add_parser(
        "selfcheck",
        help="自检所有脚本",
        description="自检所有脚本：编译检查和 --help 检查",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    selfcheck_parser.add_argument("--timeout", type=float, default=30.0, help="单脚本超时秒数（默认: 30.0）")

    return parser


def _run_topsis(args: argparse.Namespace) -> None:
    started_at = datetime.now()
    error_message: Optional[str] = None
    return_code: int = 0

    with run_context(args.run_dir) as (run_dir, log_path, stdout_buffer, stderr_buffer):
        try:
            input_file = require_input_path(args.input_file, "输入 csv 和 excel 文件", allowed_suffixes={".xlsx", ".xls", ".csv"})
            output_file = prepare_output_path(args.output_file, "entropy_topsis.xlsx")

            df = load_data(input_file)
            df.columns = df.columns.map(lambda x: str(x).strip())

            # Normalize group_cols / year_col
            year_col, group_cols = _normalize_group_year_cols(
                args.year_col, args.group_cols or "", args.group_col
            )

            # Parse negative indicators and ID columns
            negative_text = args.negative_indicators or args.negative
            negative_indicators = _parse_csv_list(negative_text) if negative_text else None
            id_cols = _parse_csv_list(args.id_cols) if args.id_cols else None

            # Run entropy TOPSIS
            result = run_entropy_topsis(
                df,
                id_cols=id_cols,
                group_cols=group_cols,
                year_col=year_col,
                negative_cols=negative_indicators,
                eps_shift=args.eps_shift,
            )

            # Write output
            entropy_df = result.entropy
            if args.append_weights and result.weights is not None and not result.weights.empty:
                if len(result.weights) == 1:
                    weights_series = result.weights.iloc[0]
                    id_cols = [entropy_df.columns[0]] if not entropy_df.empty else []
                    entropy_df = append_weights_row(entropy_df, id_cols=id_cols, weights=weights_series)

            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                entropy_df.to_excel(writer, sheet_name="熵权", index=False)
                result.topsis.to_excel(writer, sheet_name="topsis", index=False)

            print(f"分析完成，结果已写入 {output_file}（sheet: 熵权, topsis）。")
            print("\nTOPSIS 得分:")
            print(result.topsis)

        except Exception as exc:
            return_code = 1
            error_message = str(exc)
            print(f"执行失败: {exc}", file=sys.stderr)
            raise

    finished_at = datetime.now()
    stdout_text = "".join(stdout_buffer)
    stderr_text = "".join(stderr_buffer)

    write_run_summary(
        run_dir,
        tool_label="熵权TOPSIS",
        command=[sys.executable, __file__, "topsis"] + sys.argv[3:],
        return_code=return_code if return_code != 0 else 0,
        started_at=started_at,
        finished_at=finished_at,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        error_message=error_message,
    )


def _run_ranktest(args: argparse.Namespace) -> None:
    started_at = datetime.now()
    error_message: Optional[str] = None
    return_code: int = 0

    with run_context(args.run_dir) as (run_dir, log_path, stdout_buffer, stderr_buffer):
        try:
            input_file = require_input_path(args.input_file, "输入 csv 和 excel 文件", allowed_suffixes={".xlsx", ".xls", ".csv"})
            output_file = prepare_output_path(args.output_file, "rank_tests.xlsx")

            df = load_data(input_file)

            mode = (args.mode or "").strip().lower()
            if mode == "questions":
                question_cols = _parse_csv_list(args.question_cols)
                if not question_cols:
                    raise ValueError("mode=questions 时必须指定 --question-cols，例如 --question-cols Q1,Q2,Q3")
                result_df = run_rank_tests(
                    df,
                    mode="questions",
                    question_cols=question_cols,
                    dv_col=args.dv_col,
                    alpha=float(args.alpha),
                )
            elif mode == "group":
                if (args.question_cols or "").strip():
                    raise ValueError("mode=group 时不应提供 --question-cols")

                # Handle group_cols vs group-col with backward compatibility
                _, final_group_cols = _normalize_group_year_cols(
                    None, args.group_cols or "", args.group_col
                )
                if (args.group_cols or "").strip() and (args.group_col or "").strip():
                    print("注意：同时提供了 --group-cols 和 --group-col，将使用 --group-cols", file=sys.stderr)

                result_df = run_rank_tests(
                    df,
                    mode="group",
                    group_cols=final_group_cols,
                    alpha=float(args.alpha),
                )
            else:
                raise ValueError(f"未知 mode：{args.mode}")

            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                result_df.to_excel(writer, sheet_name="Summary", index=False)
            print(f"分析完成，结果已写入 {output_file}（sheet: Summary）。")
            print(result_df)

        except Exception as exc:
            return_code = 1
            error_message = str(exc)
            print(f"执行失败: {exc}", file=sys.stderr)
            raise

    finished_at = datetime.now()
    stdout_text = "".join(stdout_buffer)
    stderr_text = "".join(stderr_buffer)

    write_run_summary(
        run_dir,
        tool_label="秩和检验",
        command=[sys.executable, __file__, "ranktest"] + sys.argv[3:],
        return_code=return_code if return_code != 0 else 0,
        started_at=started_at,
        finished_at=finished_at,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        error_message=error_message,
    )


def _run_selfcheck(args: argparse.Namespace) -> int:
    started_at = datetime.now()
    error_message: Optional[str] = None
    return_code: int = 0
    final_stdout = ""
    final_stderr = ""

    try:
        with run_context() as (run_dir, log_path, stdout_buffer, stderr_buffer):
            try:
                results: list[str] = []
                all_ok = True
                project_root = Path(__file__).resolve().parents[1]
                env = os.environ.copy()
                env.setdefault("PYTHONUTF8", "1")
                env.setdefault("PYTHONIOENCODING", "utf-8")
                existing_pythonpath = env.get("PYTHONPATH", "")
                if existing_pythonpath:
                    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}"
                else:
                    env["PYTHONPATH"] = str(project_root)

                print("[self-check] Starting self-check...", flush=True)
                py_result = run_command_capture([sys.executable, "-c", "print('ok')"], env=env, timeout=args.timeout)
                py_ok = py_result.returncode == 0 and "ok" in (py_result.stdout or "")
                results.append(f"- Python: {'OK' if py_ok else 'FAIL'}")
                all_ok = all_ok and py_ok

                scripts = sorted(project_root.glob("*.py"), key=lambda p: p.name.lower())
                skip_help_names = {
                    "comprehensive_grouping_test.py",
                    "debug_grouping.py",
                    "test_grouping.py",
                    "run_tests.py",
                }
                for script_path in scripts:
                    compile_result = run_command_capture(
                        [sys.executable, "-m", "py_compile", str(script_path)],
                        env=env,
                        timeout=args.timeout,
                    )
                    compile_ok = compile_result.returncode == 0

                    name = script_path.name
                    skip_help = (
                        name in skip_help_names
                        or name.startswith("test_")
                        or name.endswith("_test.py")
                        or name.endswith("_tests.py")
                        or name.startswith("debug_")
                    )
                    if skip_help:
                        help_ok = True
                        help_status = "SKIP"
                    else:
                        help_result = run_command_capture(
                            [sys.executable, str(script_path), "--help"],
                            env=env,
                            timeout=args.timeout,
                        )
                        help_ok = help_result.returncode == 0
                        help_status = "OK" if help_ok else "FAIL"

                    results.append(
                        f"- {script_path.name}: py_compile={'OK' if compile_ok else 'FAIL'}, --help={help_status}"
                    )
                    all_ok = all_ok and compile_ok and help_ok

                status = "自检通过" if all_ok else "自检失败"
                print(status, flush=True)
                print("\n".join(results), flush=True)
                return_code = 0 if all_ok else 1

            except Exception as exc:
                return_code = 1
                error_message = str(exc)
                print(f"执行失败: {exc}", file=sys.stderr)
                raise
            finally:
                final_stdout = "".join(stdout_buffer)
                final_stderr = "".join(stderr_buffer)
    finally:
        finished_at = datetime.now()
        write_run_summary(
            run_dir,
            tool_label="自检",
            command=[sys.executable, __file__, "selfcheck"],
            return_code=return_code,
            started_at=started_at,
            finished_at=finished_at,
            stdout_text=final_stdout,
            stderr_text=final_stderr,
            error_message=error_message,
        )
    return return_code


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "topsis":
            _run_topsis(args)
        elif args.command == "ranktest":
            _run_ranktest(args)
        elif args.command == "selfcheck":
            return _run_selfcheck(args)
        else:
            raise ValueError(f"未知命令：{args.command}")
    except Exception as exc:  # noqa: BLE001 - CLI should not explode with stacktrace
        print(f"执行失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
