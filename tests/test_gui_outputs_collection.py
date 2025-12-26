import os
import zipfile
from pathlib import Path
import pytest
from gradio_toolbox import collect_artifacts

def test_collect_artifacts_basic(tmp_path):
    # Setup: Create a mock run directory
    run_dir = tmp_path / "run_123"
    run_dir.mkdir()
    
    # Create some output files
    (run_dir / "output.xlsx").write_text("dummy excel content")
    (run_dir / "summary.md").write_text("dummy summary content")
    (run_dir / "run.log").write_text("dummy log content")
    
    # Create an image
    (run_dir / "plot.png").write_text("dummy image content")
    
    # Create a file in inputs (should be ignored)
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "input.xlsx").write_text("dummy input content")
    
    # Create a hidden file (should be ignored)
    (run_dir / ".hidden").write_text("hidden")
    
    # Run collection
    images_with_captions, download_files = collect_artifacts(run_dir)
    
    # Assertions
    # 1. Images included in gallery
    assert len(images_with_captions) == 1
    assert Path(images_with_captions[0][0]).name == "plot.png"
    assert images_with_captions[0][1] == "plot.png"
    
    # 2. All expected files in download list (including image, summary, log, and zip)
    # Total expected: output.xlsx, summary.md, run.log, plot.png -> 4 files -> zip should be generated
    # Total files in list should be 5
    download_names = [Path(p).name for p in download_files]
    assert "output.xlsx" in download_names
    assert "summary.md" in download_names
    assert "run.log" in download_names
    assert "plot.png" in download_names
    assert "all_outputs.zip" in download_names
    
    # 3. Zip content matches list (minus the zip itself)
    zip_path = run_dir / "all_outputs.zip"
    assert zip_path.exists()
    
    with zipfile.ZipFile(zip_path, "r") as z:
        zip_contents = z.namelist()
        assert "output.xlsx" in zip_contents
        assert "summary.md" in zip_contents
        assert "run.log" in zip_contents
        assert "plot.png" in zip_contents
        assert "all_outputs.zip" not in zip_contents
        assert "inputs/input.xlsx" not in zip_contents
        assert ".hidden" not in zip_contents

def test_collect_artifacts_single_file_no_zip(tmp_path):
    run_dir = tmp_path / "run_single"
    run_dir.mkdir()
    (run_dir / "only_one.txt").write_text("just one")
    
    images_with_captions, download_files = collect_artifacts(run_dir)
    
    assert len(images_with_captions) == 0
    assert len(download_files) == 1
    assert Path(download_files[0]).name == "only_one.txt"
    assert not (run_dir / "all_outputs.zip").exists()

def test_collect_artifacts_nested_directories(tmp_path):
    run_dir = tmp_path / "run_nested"
    run_dir.mkdir()
    
    (run_dir / "top.txt").write_text("top")
    sub_dir = run_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "bottom.png").write_text("bottom image")
    
    images_with_captions, download_files = collect_artifacts(run_dir)
    
    download_rel_names = [Path(p).relative_to(run_dir).as_posix() for p in download_files]
    assert "top.txt" in download_rel_names
    assert "sub/bottom.png" in download_rel_names
    
    zip_path = run_dir / "all_outputs.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as z:
        # Zip namelist always uses forward slashes per spec
        zip_contents = z.namelist()
        assert "top.txt" in zip_contents
        assert "sub/bottom.png" in zip_contents

def test_collect_artifacts_deduplication(tmp_path):
    # Test that images aren't duplicated in download_files if they are already there
    # (Although collect_artifacts logic seems to handle it with seen_files)
    run_dir = tmp_path / "run_dedup"
    run_dir.mkdir()
    (run_dir / "image.png").write_text("image")
    (run_dir / "data.csv").write_text("data")
    
    images_with_captions, download_files = collect_artifacts(run_dir)
    
    download_names = [Path(p).name for p in download_files]
    # image.png, data.csv, all_outputs.zip
    assert len(download_files) == 3
    assert download_names.count("image.png") == 1
