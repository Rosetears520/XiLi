import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def parse_threshold(value: int, default: int = 128) -> int:
    try:
        threshold = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(255, threshold))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="黑白图片转换工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入图片路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出图片路径（默认: <输入文件名>_binary.<后缀>）",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="阈值（0-255，默认: 128）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(input_file: Path, output_file: Path, threshold: int) -> None:
    img = Image.open(input_file).convert("L")
    arr = np.array(img)
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)

    if output_file.parent:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(output_file)


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        src = require_input_path(
            args.input_file,
            "输入图片",
            allowed_suffixes={".png", ".jpg", ".jpeg", ".bmp", ".webp"},
        )
        if args.output_file:
            output_file = prepare_output_path(args.output_file, src.with_name("binary.png").name)
        else:
            suffix = src.suffix if src.suffix else ".png"
            output_file = prepare_output_path(None, f"{src.stem}_binary{suffix}")

        threshold = parse_threshold(args.threshold)
        main(input_file=src, output_file=output_file, threshold=threshold)
    except Exception as exc:
        print(f"转换失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
