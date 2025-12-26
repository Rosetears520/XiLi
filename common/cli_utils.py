from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

def get_project_root() -> Path:
    """Get the root directory of the project, handling PyInstaller frozen state."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller bundled
        return Path(sys._MEIPASS)
    # Normal Python script
    return Path(__file__).resolve().parents[1]

PROJECT_ROOT = get_project_root()
RUNS_ROOT = PROJECT_ROOT / "runs"


def create_run_dir(run_dir: Optional[str] = None) -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = (PROJECT_ROOT / run_path).resolve()
        run_path.mkdir(parents=True, exist_ok=True)
    else:
        cwd = Path.cwd()
        try:
            cwd.relative_to(RUNS_ROOT)
            run_path = cwd
        except ValueError:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            suffix = uuid.uuid4().hex[:6]
            run_path = RUNS_ROOT / f"{timestamp}_{suffix}"
            run_path.mkdir(parents=True, exist_ok=True)

    (run_path / "inputs").mkdir(exist_ok=True)
    return run_path


def enter_run_dir(run_dir: Optional[str] = None) -> Path:
    run_path = create_run_dir(run_dir)
    os.chdir(run_path)
    return run_path


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path)
    if cwd_path.exists():
        return cwd_path.resolve()

    return (PROJECT_ROOT / path).resolve()


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def require_input_path(
    path_str: Optional[str],
    label: str,
    *,
    must_exist: bool = True,
    is_dir: bool = False,
    allowed_suffixes: Optional[Iterable[str]] = None,
) -> Path:
    if not path_str:
        raise ValueError(f"{label} 不能为空，请提供有效路径。")
    resolved = resolve_input_path(path_str)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{label} 不存在：{resolved}")
    if must_exist:
        if is_dir and not resolved.is_dir():
            raise NotADirectoryError(f"{label} 不是目录：{resolved}")
        if not is_dir and not resolved.is_file():
            raise FileNotFoundError(f"{label} 不是文件：{resolved}")
    if allowed_suffixes:
        suffix = resolved.suffix.lower()
        allowed = {s.lower() for s in allowed_suffixes}
        if suffix and suffix not in allowed:
            raise ValueError(f"{label} 扩展名不匹配：{resolved}，允许：{sorted(allowed)}")
    return resolved


def prepare_output_path(
    path_str: Optional[str],
    default_name: str,
    *,
    ensure_parent: bool = True,
) -> Path:
    if path_str:
        resolved = resolve_output_path(path_str)
    else:
        resolved = (Path.cwd() / default_name).resolve()
    if ensure_parent and resolved.parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def stream_process_output(
    stream: Optional[object],
    buffer: list[str],
    *,
    to_stderr: bool,
) -> None:
    if stream is None:
        return
    # Use readline for real-time streaming to console
    for line in iter(getattr(stream, "readline", lambda: ""), ""):
        buffer.append(line)
        if to_stderr:
            print(line, end="", file=sys.stderr, flush=True)
        else:
            print(line, end="", flush=True)
    try:
        getattr(stream, "close", lambda: None)()
    except Exception:
        pass


def run_command_capture(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    command_list = [str(item) for item in command]
    print(f"[self-check] Running: {subprocess.list2cmdline(command_list)}", flush=True)
    process = subprocess.Popen(
        command_list,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_thread = threading.Thread(
        target=stream_process_output,
        args=(process.stdout, stdout_lines),
        kwargs={"to_stderr": False},
    )
    stderr_thread = threading.Thread(
        target=stream_process_output,
        args=(process.stderr, stderr_lines),
        kwargs={"to_stderr": True},
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[self-check] Timeout after {timeout} seconds", file=sys.stderr, flush=True)
        process.kill()
        return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)
    print(f"[self-check] Exit code: {return_code}", flush=True)
    return subprocess.CompletedProcess(command_list, return_code, stdout_text, stderr_text)


class _TeeWriter:
    def __init__(
        self,
        stream: TextIO,
        log_fp: TextIO,
        lock: threading.Lock,
        buffer: list[str],
    ) -> None:
        self._stream = stream
        self._log_fp = log_fp
        self._lock = lock
        self._buffer = buffer

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._stream.write(text)
            self._stream.flush()
            self._log_fp.write(text)
            self._log_fp.flush()
        self._buffer.append(text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()
            self._log_fp.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@contextmanager
def tee_console_to_log(run_dir: Path):
    log_path = run_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stdout_buffer: list[str] = []
    stderr_buffer: list[str] = []
    with log_path.open("w", encoding="utf-8-sig") as log_fp:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeWriter(original_stdout, log_fp, lock, stdout_buffer)
        sys.stderr = _TeeWriter(original_stderr, log_fp, lock, stderr_buffer)
        try:
            yield stdout_buffer, stderr_buffer, log_path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
