import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
    """
    Resolve font path from input (either a path or a font family name).
    """
    if not font_input:
        return resolve_default_font_path()
    
    # If it's an existing file, use it directly
    if os.path.exists(font_input) and os.path.isfile(font_input):
        return font_input
    
    # Try to find font by family name
    try:
        font_path = font_manager.findfont(font_input, fallback_to_default=False)
        if font_path and os.path.exists(font_path):
            return font_path
    except Exception:
        pass
        
    return None


def gradient_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

def fixed_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """
    固定颜色函数示例：
    返回固定颜色（例如蓝色）。
    """
    return "rgb(0, 0, 255)"

def generate_wordcloud(
    freq_file,
    output_image,
    background_color='white',
    width=800,
    height=600,
    scale=1,                 # 按比例放大画布，默认为1，若设为1.5，则宽高均为原来的1.5倍
    mask_image_path=None,
    font_path=None,
    dpi=300,
    prefer_horizontal=0.90,  # 词语水平方向排版出现的频率，默认 0.90
    min_font_size=4,         # 最小字体大小，默认 4
    font_step=1,             # 字体步长，默认 1
    max_words=200,           # 显示的最大词数，默认 200
    max_font_size=None,      # 最大字体大小，默认 None（自动调整）
    relative_scaling=0.5,    # 词频与字体大小的关联性，默认 0.5，数值越大字体大小根据词频差异越明显
    color_func=None,         # 自定义颜色生成函数，默认为 None；当传入 "gradient" 时使用渐变色函数，
                             # 传入 "fixed" 时使用固定颜色函数；若为 None，则不使用自定义颜色函数。
    colormap="viridis"       # colormap 名称，默认 "viridis"，当 color_func 为 None 时生效
):
    """
    根据词频文件生成词云图，并保存为图片。

    参数说明：
    - freq_file: 词频文件路径（txt 格式，UTF-8 编码），每行格式为：<word> <frequency>，以空白符分隔。
    - output_image: 输出词云图片的文件路径（如 'wordcloud.png'）。
    - background_color: 词云背景颜色，默认为 'white'。
    - width: 词云图片宽度（像素），默认为 800。
    - height: 词云图片高度（像素），默认为 600。
    - scale: 按比例放大画布，如设置为 1.5，则生成的图片宽高分别为 width*1.5 和 height*1.5。
    - mask_image_path: 词云遮罩图片路径，默认为 None，若指定则使用该图片作为词云形状。
    - font_path: 字体文件路径（如 'C:\\Windows\\Fonts\\SimHei.ttf'），未指定时将尝试使用系统中文字体（如微软雅黑）。
    - dpi: 输出图片的 DPI（每英寸点数），默认为 300。
    - prefer_horizontal: 词语水平方向排版出现的频率，默认 0.90。
    - min_font_size: 显示的最小字体大小，默认 4。
    - font_step: 字体步长，默认 1。
    - max_words: 要显示的最大词数，默认 200。
    - max_font_size: 显示的最大字体大小，默认 None（自动调整）。
    - relative_scaling: 词频与字体大小的关联性，默认 0.5。
    - color_func: 自定义颜色生成函数。若传入 "gradient" 则使用渐变色函数，传入 "fixed" 则使用固定颜色函数，
                  若为 None，则不使用自定义颜色函数（此时使用 colormap 生成颜色）。
    - colormap: 使用的 colormap 名称或 matplotlib colormap 对象，默认 "viridis"，当 color_func 为 None 时生效。

    注意：
    当使用遮罩图片时，词云图形状通常由遮罩图片尺寸决定，
    因此 width、height 和 scale 对最终输出尺寸影响可能有限，如需调整请修改遮罩图片或后处理。
    """
    # 如果 color_func 是字符串，根据输入选择对应的函数
    if isinstance(color_func, str):
        if color_func.lower() == "gradient":
            color_func = gradient_color_func
        elif color_func.lower() == "fixed":
            color_func = fixed_color_func
        else:
            color_func = None

    if not font_path:
        font_path = resolve_default_font_path()
        if font_path:
            print(f"未指定字体，使用默认字体: {font_path}")
        else:
            print("未指定字体，未找到可用中文字体，将使用默认字体（可能出现方框）。")

    # 读取词频文件，构建词频字典
    word_freq = {}
    with open(freq_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    freq = int(parts[1])
                except ValueError:
                    try:
                        freq = float(parts[1])
                    except Exception:
                        freq = 1
                word_freq[word] = freq

    # 如果指定了遮罩图片，则检查路径并加载
    mask = None
    if mask_image_path:
        if os.path.exists(mask_image_path):
            mask = imageio.imread(mask_image_path)
            print(f"已成功加载遮罩图片，形状为 {mask.shape}")
        else:
            print(f"遮罩图片文件 {mask_image_path} 不存在，忽略遮罩设置。")

    # 根据 scale 调整实际画布尺寸
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    # 创建 WordCloud 对象，设置各项参数
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
        colormap=colormap
    )

    # 根据词频生成词云
    wc.generate_from_frequencies(word_freq)

    # 获取 PIL Image 对象，并以指定 dpi 保存图片
    wc_image = wc.to_image()
    wc_image.save(output_image, dpi=(dpi, dpi))
    print(f"词云图片已保存到 {output_image} (dpi={dpi})")

    # 可选：展示词云图片（如需展示则取消注释）
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

# =============== 示例调用 ===============
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
        help="字体文件路径（可选，未指定时尝试使用系统中文字体）",
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
    font_path: Optional[Path],
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
        freq_file=str(input_file),
        output_image=str(output_image),
        background_color=background_color,
        width=width,
        height=height,
        scale=scale,
        mask_image_path=str(mask_image_path) if mask_image_path else None,
        font_path=str(font_path) if font_path else None,
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
                allowed_suffixes={ ".png", ".jpg", ".jpeg", ".bmp", ".webp" },
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

"""
第一次使用需要安装第三方库:pip install imageio wordcloud
附注说明：
- freq_file: 每行格式要求为 "<word> <frequency>"，以空格分隔。
- output_image: 指定生成的词云图片保存路径（支持 png、jpg 等格式）。
- background_color: 词云背景颜色。
- width, height: 原始画布尺寸（像素），当使用遮罩图片时，这些参数对最终形状影响较小。
- scale: 画布放大倍数。例如，scale=1.5 表示实际输出图片宽度和高度分别为原始宽度和高度的 1.5 倍。
- mask_image_path: 遮罩图片路径，用于确定词云形状；如不指定则使用矩形形状。
- font_path: 用于显示中文的字体路径；未指定时将尝试使用系统中文字体（如微软雅黑）。
- dpi: 输出图片的分辨率（每英寸点数）。
- prefer_horizontal: 控制词语水平方向排版的概率（0 到 1 之间）。
- min_font_size: 显示的最小字体大小。
- font_step: 字体步长，较大步长可加快生成速度，但可能降低细节。
- max_words: 词云中最多显示的单词数量。
- max_font_size: 显示的最大字体大小；若为 None 则自动调整。
- relative_scaling: 词频与字体大小关联性，数值越大词频差异越明显。
- color_func: 自定义颜色函数设置，可传入 "gradient"（渐变色）、"fixed"（固定颜色）或 None（默认使用 colormap）。
- colormap: 当 color_func 为 None 时，WordCloud 会使用此 colormap 为单词随机分配颜色。
    注意：
    当使用遮罩图片时，词云图形状通常由遮罩图片的尺寸决定，
    因此 width、height 和 scale 对最终输出尺寸的影响可能有限，如需调整请修改遮罩图片或后处理。
"""
