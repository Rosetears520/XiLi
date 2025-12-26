import os
import sys
from pathlib import Path
import pytest
from xili.cli import cli
import common.cli_utils

def test_selfcheck_logging(tmp_path, monkeypatch, capsys):
    # 1. Redirect RUNS_ROOT to tmp_path
    monkeypatch.setattr(common.cli_utils, "RUNS_ROOT", tmp_path)
    
    # 2. Mock the glob to only check a few files to speed up the test
    # We need to find where Path comes from in xili.cli
    # It's from pathlib.Path
    
    original_glob = Path.glob
    def mocked_glob(self, pattern):
        if pattern == "*.py" and "XiLiSuite" in str(self):
            # Return only a few representative files
            return [
                self / "xili" / "cli.py",
                self / "common" / "cli_utils.py"
            ]
        return original_glob(self, pattern)
    
    # This might be tricky because Path is a class and glob is an instance method
    # Monkeypatching Path.glob directly
    monkeypatch.setattr(Path, "glob", mocked_glob)

    old_cwd = os.getcwd()
    try:
        # 3. Run selfcheck
        # Note: --timeout is for each command
        exit_code = cli(["selfcheck", "--timeout", "10.0"])
        assert exit_code == 0
    finally:
        os.chdir(old_cwd)

    # 4. Check stdout
    captured = capsys.readouterr()
    # Note: tee_console_to_log will have printed to Pytest's captured stdout
    assert "[self-check] Starting self-check..." in captured.out
    assert "自检通过" in captured.out
    
    # 5. Find run directory
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    
    # 6. Verify files
    run_log = run_dir / "run.log"
    summary_md = run_dir / "summary.md"
    assert run_log.exists()
    assert summary_md.exists()
    
    log_content = run_log.read_text(encoding="utf-8-sig")
    summary_content = summary_md.read_text(encoding="utf-8-sig")
    
    # 7. Verify log completeness in run.log
    assert "[self-check] Starting self-check..." in log_content
    assert "自检通过" in log_content
    
    # 8. Verify summary embeds the log
    assert "[stdout]" in summary_content
    assert "[self-check] Starting self-check..." in summary_content
    assert "自检通过" in summary_content
    
    # Verify that run_command_capture output is also present
    assert "[self-check] Running:" in log_content
    assert "[self-check] Exit code: 0" in log_content
