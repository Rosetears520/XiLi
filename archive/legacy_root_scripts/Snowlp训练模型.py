import argparse
import sys
from pathlib import Path
from typing import Optional

from snownlp import sentiment

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SnowNLP 情感模型训练工具")
    parser.add_argument(
        "--negative-file",
        default=None,
        help="负面语料文件路径",
    )
    parser.add_argument(
        "--positive-file",
        default=None,
        help="正面语料文件路径",
    )
    parser.add_argument(
        "--output-model",
        default=None,
        help="输出模型文件路径（默认: sentiment.marshal）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(negative_file: Path, positive_file: Path, output_model: Path) -> None:
    sentiment.train(str(negative_file), str(positive_file))
    sentiment.save(str(output_model))


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        negative_file = require_input_path(
            args.negative_file,
            "负面语料文件",
            allowed_suffixes={".txt"},
        )
        positive_file = require_input_path(
            args.positive_file,
            "正面语料文件",
            allowed_suffixes={".txt"},
        )
        output_model = prepare_output_path(args.output_model, "sentiment.marshal")

        main(
            negative_file=negative_file,
            positive_file=positive_file,
            output_model=output_model,
        )
    except Exception as exc:
        print(f"训练失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
