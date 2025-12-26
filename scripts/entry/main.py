from __future__ import annotations

import runpy
import sys
from typing import Callable

from xili.cli import cli as xili_cli

_CORE_COMMANDS = {"topsis", "ranktest", "selfcheck"}
_TOOL_COMMANDS = {
    "wordcloud": "word_cloud",
    "word_segmentation": "word_segmentation",
    "binary_image": "binary_image",
    "linear_interpolation": "linear_interpolation",
    "yearly_average": "yearly_average",
    "growth_rate": "average_growth_rate",
    "kmeans": "kmeans",
    "lda_eval": "lda_evaluation",
    "lda_model": "lda_model",
    "snowlp_train": "snownlp_train",
    "snowlp_analysis": "snownlp_sentiment",
}

def _run_module(module_name: str, argv: list[str]) -> int:
    # In Nuitka standalone builds, `runpy.run_module` may fail because Nuitka's
    # loader does not implement `get_code`, which runpy expects.
    #
    # Prefer calling a module-level `cli(argv)` entrypoint when available.
    try:
        import importlib

        mod = importlib.import_module(module_name)
        cli = getattr(mod, "cli", None)
        if callable(cli):
            return int(cli(argv))
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    except Exception:
        pass

    original_argv = sys.argv[:]
    try:
        sys.argv = [module_name] + argv
        runpy.run_module(module_name, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = original_argv
    return 0


def _run_gui(argv: list[str]) -> int:
    try:
        from scripts.gui import gradio_toolbox
    except Exception as exc:  # noqa: BLE001 - CLI should not explode with stacktrace
        print(f"GUI 启动失败: {exc}", file=sys.stderr)
        return 1

    launcher: Callable[[list[str] | None], int] | None = getattr(gradio_toolbox, "main", None)
    if launcher is None:
        print("GUI 入口缺失：scripts.gui.gradio_toolbox.main 未定义。", file=sys.stderr)
        return 1
    return launcher(argv)


def _print_usage() -> None:
    commands = sorted(list(_CORE_COMMANDS) + ["gui"] + list(_TOOL_COMMANDS.keys()))
    prog = sys.argv[0] if sys.argv else "main.py"
    print(f"Usage: {prog} <command> [options]", file=sys.stderr)
    print("Commands:", ", ".join(commands), file=sys.stderr)
    print("Tip: gui 启动 GUI；或用 topsis/ranktest/selfcheck 运行核心 CLI。", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        return _run_gui([])

    command = args[0]
    if command in {"-h", "--help", "help"}:
        _print_usage()
        return 0
    if command in _CORE_COMMANDS:
        return xili_cli(args)
    if command == "gui":
        return _run_gui(args[1:])

    module_name = _TOOL_COMMANDS.get(command)
    if module_name:
        return _run_module(module_name, args[1:])

    _print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
