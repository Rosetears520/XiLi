from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from common.cli_utils import enter_run_dir, tee_console_to_log

@contextmanager
def run_context(run_dir_arg: Optional[str] = None):
    """
    Unified context for CLI tools:
    1. Enters a run directory.
    2. Tees console output to run.log with utf-8-sig.
    """
    run_dir = enter_run_dir(run_dir_arg)
    with tee_console_to_log(run_dir) as (stdout_buffer, stderr_buffer, log_path):
        yield run_dir, log_path, stdout_buffer, stderr_buffer
