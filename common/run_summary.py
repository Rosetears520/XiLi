from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

def _format_command(command: List[str]) -> str:
    if not command:
        return ""
    return subprocess.list2cmdline(command)


def _collect_files(
    run_dir: Path,
    *,
    skip_dirs: Iterable[str],
    skip_names: Iterable[str],
    skip_suffixes: Iterable[str],
) -> List[str]:
    results: List[str] = []
    skip_dir_set = set(skip_dirs)
    skip_name_set = set(skip_names)
    skip_suffix_set = {suffix.lower() for suffix in skip_suffixes}

    for fp in sorted(run_dir.rglob("*")):
        if not fp.is_file():
            continue
        if fp.name.startswith("."):
            continue
        if fp.name in skip_name_set:
            continue
        if fp.suffix.lower() in skip_suffix_set:
            continue
        if any(skip_dir in fp.parts for skip_dir in skip_dir_set):
            continue
        results.append(str(fp.relative_to(run_dir)))
    return results


def collect_input_files(run_dir: Path) -> List[str]:
    inputs_dir = run_dir / "inputs"
    if not inputs_dir.exists():
        return []
    results: List[str] = []
    for fp in sorted(inputs_dir.rglob("*")):
        if not fp.is_file():
            continue
        if fp.name.startswith("."):
            continue
        if fp.suffix.lower() in {".pyc", ".pyo"}:
            continue
        results.append(str(fp.relative_to(run_dir)))
    return results


def collect_output_files(run_dir: Path) -> List[str]:
    return _collect_files(
        run_dir,
        skip_dirs=["inputs"],
        skip_names=[],
        skip_suffixes=[".pyc", ".pyo"],
    )


def write_run_summary(
    run_dir: Path,
    *,
    tool_label: str,
    command: List[str],
    return_code: Optional[int],
    started_at: datetime,
    finished_at: datetime,
    stdout_text: str = "",
    stderr_text: str = "",
    error_message: Optional[str] = None,
) -> Path:
    duration_seconds = max(0.0, (finished_at - started_at).total_seconds())
    command_text = _format_command(command)

    input_files = collect_input_files(run_dir)
    output_files = collect_output_files(run_dir)

    lines: List[str] = [
        f"# 数据处理总结 - {tool_label}",
        "",
        "## 运行环境",
        f"- **工具**：{tool_label}",
        f"- **运行目录**：`{run_dir}`",
        f"- **开始时间**：{started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **结束时间**：{finished_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **耗时**：{duration_seconds:.1f} 秒",
        f"- **返回码**：`{return_code if return_code is not None else 'N/A'}`",
        "",
        "## 执行命令",
        "```bash",
        command_text,
        "```",
        "",
        "## 文件列表",
        "",
        "### 输入文件",
    ]

    if input_files:
        lines.extend([f"- `{path}`" for path in input_files])
    else:
        lines.append("- （无）")

    lines.append("")
    lines.append("### 输出文件")
    if output_files:
        lines.extend([f"- `{path}`" for path in output_files])
    else:
        lines.append("- （无）")

    if error_message:
        lines.append("")
        lines.append("## 异常/错误")
        lines.append(f"> ❌ {error_message}")

    if stdout_text or stderr_text:
        lines.append("")
        lines.append("## 运行日志")
        if stdout_text:
            lines.append("### [stdout]")
            lines.append("```text")
            lines.append(stdout_text)
            lines.append("```")
        if stderr_text:
            lines.append("### [stderr]")
            lines.append("```text")
            lines.append(stderr_text)
            lines.append("```")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return summary_path
