from __future__ import annotations

import runpy
from pathlib import Path
from typing import Callable, Optional


def _load_legacy_cli() -> Callable[[Optional[list[str]]], int]:
    impl_path = Path(__file__).resolve().parents[2] / "K-means.py"
    if not impl_path.exists():
        raise FileNotFoundError(f"缺少实现脚本：{impl_path}")

    module_globals = runpy.run_path(str(impl_path), run_name="kmeans_impl")
    cli_func = module_globals.get("cli")
    if not callable(cli_func):
        raise RuntimeError("K-means.py 未暴露 cli(argv) 函数，无法委托执行。")
    return cli_func  # type: ignore[return-value]


def cli(argv: Optional[list[str]] = None) -> int:
    return int(_load_legacy_cli()(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
