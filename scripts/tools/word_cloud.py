from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import cm, font_manager
from PIL import Image
from wordcloud import WordCloud

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path

DEFAULT_FONT_PATH_CANDIDATES = [
    r"C:\\Windows\\Fonts\\msyh.ttc",
    r"C:\\Windows\\Fonts\\msyh.ttf",
    r"C:\\Windows\\Fonts\\simhei.ttf",
    r"C:\\Windows\\Fonts\\simkai.ttf",
]
DEFAULT_FONT_FAMILIES = [
    "Microsoft YaHei",
    "SimHei",
    "KaiTi",
    "SimSun",
]


def resolve_default_font_path() -> Optional[str]:
    for candidate in DEFAULT_FONT_PATH_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    for family in DEFAULT_FONT_FAMILIES:
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
        except Exception:
            continue
        if font_path and os.path.exists(font_path):
            return font_path
    return None


def resolve_font_path(font_input: Optional[str]) -> Optional[str]:
    if not font_input:
        return resolve_default_font_path()

    if os.path.exists(font_input) and os.path.isfile(font_input):
        return font_input

    try:
        font_path = font_manager.findfont(font_input, fallback_to_default=False)
        if font_path and os.path.exists(font_path):
            return font_path
    except Exception:
        pass

    return None


def _rgb_text(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, int(channel * 255))) for channel in rgb)
    return f"rgb({r}, {g}, {b})"


def _gradient_color_func(colormap_name: str, min_font_size: int, max_font_size: Optional[int]):
    cmap = cm.get_cmap(colormap_name)
    max_font = max_font_size if max_font_size is not None else min_font_size + 60
    span = max(1, max_font - min_font_size)

    def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        ratio = (font_size - min_font_size) / span
        ratio = min(1.0, max(0.0, ratio))
        color = cmap(ratio)[:3]
        return _rgb_text(color)

    return _color_func


def _fixed_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "rgb(0, 0, 255)"


def _load_word_frequencies(freq_file: Path) -> dict[str, float]:
    word_freq: dict[str, float] = {}
    with freq_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    freq = float(parts[1])
                except ValueError:
                    freq = 1.0
                word_freq[word] = freq
    return word_freq


def _load_mask(mask_image_path: Optional[Path]) -> Optional[np.ndarray]:
    if not mask_image_path:
        return None
    if not mask_image_path.exists():
        raise FileNotFoundError(f"遮罩图片不存在: {mask_image_path}")
    mask_image = Image.open(mask_image_path).convert("L")
    return np.array(mask_image)


def generate_wordcloud(
    freq_file: Path,
    output_image: Path,
    background_color: str = "white",
    width: int = 1600,
    height: int = 900,
    scale: float = 1.0,
    mask_image_path: Optional[Path] = None,
    font_path: Optional[str] = None,
    dpi: int = 300,
    prefer_horizontal: float = 1.0,
    min_font_size: int = 4,
    font_step: int = 1,
    max_words: int = 500,
    max_font_size: Optional[int] = None,
    relative_scaling: float = 0.5,
    color_func: Optional[str] = None,
    colormap: str = "viridis",
) -> None:
    if isinstance(color_func, str):
        if color_func.lower() == "gradient":
            color_func = _gradient_color_func(colormap, min_font_size, max_font_size)
        elif color_func.lower() == "fixed":
            color_func = _fixed_color_func
        elif color_func.lower() in {"none", ""}:
            color_func = None

    if not font_path:
        font_path = resolve_default_font_path()
        if font_path:
            print(f"未指定字体，使用默认字体: {font_path}")
        else:
            print("未指定字体，未找到可用中文字体，将使用默认字体（可能出现方框）。")

    word_freq = _load_word_frequencies(freq_file)
    if not word_freq:
        raise ValueError("词频文件为空或格式不正确，无法生成词云。")

    mask = _load_mask(mask_image_path)
    if mask is not None:
        print(f"已加载遮罩图片，形状: {mask.shape}")

    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    wc = WordCloud(
        background_color=background_color,
        width=scaled_width,
        height=scaled_height,
        mask=mask,
        font_path=font_path,
        collocations=False,
        prefer_horizontal=prefer_horizontal,
        min_font_size=min_font_size,
        font_step=font_step,
        max_words=max_words,
        max_font_size=max_font_size,
        relative_scaling=relative_scaling,
        color_func=color_func,
        colormap=colormap,
    )

    wc.generate_from_frequencies(word_freq)
    wc_image = wc.to_image()
    if output_image.parent:
        output_image.parent.mkdir(parents=True, exist_ok=True)
    wc_image.save(output_image, dpi=(dpi, dpi))
    print(f"词云图片已保存到 {output_image} (dpi={dpi})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="词云图生成工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="词频文件路径（txt）",
    )
    parser.add_argument(
        "--output-image",
        default=None,
        help="输出词云图片路径（默认: wordcloud.png）",
    )
    parser.add_argument("--background-color", default="white", help="背景颜色（默认: white）")
    parser.add_argument("--width", type=int, default=1600, help="图片宽度（默认: 1600）")
    parser.add_argument("--height", type=int, default=900, help="图片高度（默认: 900）")
    parser.add_argument("--scale", type=float, default=1.0, help="画布缩放比例（默认: 1.0）")
    parser.add_argument("--mask-image-path", default=None, help="遮罩图片路径（可选）")
    parser.add_argument(
        "--font-path",
        default=None,
        help="字体文件路径或字体名称（可选，未指定时尝试使用系统中文字体）",
    )
    parser.add_argument("--dpi", type=int, default=300, help="输出图片 DPI（默认: 300）")
    parser.add_argument(
        "--prefer-horizontal",
        type=float,
        default=1.0,
        help="词语水平排版概率（默认: 1.0）",
    )
    parser.add_argument("--min-font-size", type=int, default=4, help="最小字体大小（默认: 4）")
    parser.add_argument("--font-step", type=int, default=1, help="字体步长（默认: 1）")
    parser.add_argument("--max-words", type=int, default=500, help="最大词数（默认: 500）")
    parser.add_argument("--max-font-size", type=int, default=None, help="最大字体大小（可选）")
    parser.add_argument(
        "--relative-scaling",
        type=float,
        default=0.5,
        help="词频与字体大小关联性（默认: 0.5）",
    )
    parser.add_argument(
        "--color-func",
        default=None,
        help="自定义颜色函数（gradient/fixed/none）",
    )
    parser.add_argument("--colormap", default="viridis", help="colormap 名称（默认: viridis）")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(
    input_file: Path,
    output_image: Path,
    background_color: str,
    width: int,
    height: int,
    scale: float,
    mask_image_path: Optional[Path],
    font_path: Optional[str],
    dpi: int,
    prefer_horizontal: float,
    min_font_size: int,
    font_step: int,
    max_words: int,
    max_font_size: Optional[int],
    relative_scaling: float,
    color_func: Optional[str],
    colormap: str,
) -> None:
    generate_wordcloud(
        freq_file=input_file,
        output_image=output_image,
        background_color=background_color,
        width=width,
        height=height,
        scale=scale,
        mask_image_path=mask_image_path,
        font_path=font_path,
        dpi=dpi,
        prefer_horizontal=prefer_horizontal,
        min_font_size=min_font_size,
        font_step=font_step,
        max_words=max_words,
        max_font_size=max_font_size,
        relative_scaling=relative_scaling,
        color_func=color_func,
        colormap=colormap,
    )


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "词频文件",
            allowed_suffixes={".txt"},
        )
        output_image = prepare_output_path(args.output_image, "wordcloud.png")
        mask_image_path = (
            require_input_path(
                args.mask_image_path,
                "遮罩图片",
                must_exist=True,
                allowed_suffixes={".png", ".jpg", ".jpeg", ".bmp", ".webp"},
            )
            if args.mask_image_path
            else None
        )
        font_path = resolve_font_path(args.font_path)

        color_func = args.color_func
        if isinstance(color_func, str) and color_func.lower() == "none":
            color_func = None

        main(
            input_file=input_file,
            output_image=output_image,
            background_color=args.background_color,
            width=args.width,
            height=args.height,
            scale=args.scale,
            mask_image_path=mask_image_path,
            font_path=font_path,
            dpi=args.dpi,
            prefer_horizontal=args.prefer_horizontal,
            min_font_size=args.min_font_size,
            font_step=args.font_step,
            max_words=args.max_words,
            max_font_size=args.max_font_size,
            relative_scaling=args.relative_scaling,
            color_func=color_func,
            colormap=args.colormap,
        )
    except Exception as exc:
        print(f"生成失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
