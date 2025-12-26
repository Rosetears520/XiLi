from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import jieba
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def _resolve_font_path(font_input: Optional[str]) -> Optional[str]:
    if not font_input:
        return None
    if os.path.exists(font_input) and os.path.isfile(font_input):
        return font_input
    try:
        from matplotlib import font_manager

        font_path = font_manager.findfont(font_input, fallback_to_default=False)
        if font_path and os.path.exists(font_path):
            return font_path
    except Exception:
        return None
    return None


def _load_texts(file_path: Path) -> list[list[str]]:
    texts: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = jieba.lcut(line)
            if tokens:
                texts.append(tokens)
    return texts


def run_lda_evaluation(
    input_file: Path,
    *,
    topic_min: int,
    topic_max: int,
    passes: int,
    workers: int,
    output_excel: Path,
    output_chart: Path,
    font: str,
    font_path: Optional[str],
    headless: bool,
) -> None:
    if topic_min < 2 or topic_max < topic_min:
        raise ValueError("topic_min/topic_max 不合法：需要 topic_min>=2 且 topic_max>=topic_min。")

    texts = _load_texts(input_file)
    if not texts:
        raise ValueError("输入文本为空，无法评估。")

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    topic_range = range(topic_min, topic_max + 1)

    coherence_values: list[float] = []
    perplexity_values: list[float] = []

    for num_topics in topic_range:
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            per_word_topics=True,
            workers=workers,
        )
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v")
        coherence_values.append(float(coherence_model.get_coherence()))
        perplexity_values.append(float(model.log_perplexity(corpus)))

    data = {
        "主题数": list(topic_range),
        "一致性得分": coherence_values,
        "困惑度 (log_perplexity)": perplexity_values,
    }
    pd.DataFrame(data).to_excel(output_excel, index=False)

    import matplotlib.pyplot as plt

    if headless:
        plt.switch_backend("Agg")

    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(list(topic_range), coherence_values, marker="o")
    plt.xlabel("主题数")
    plt.ylabel("一致性得分")
    plt.title("LDA模型一致性曲线")

    plt.subplot(1, 2, 2)
    plt.plot(list(topic_range), perplexity_values, marker="o")
    plt.xlabel("主题数")
    plt.ylabel("困惑度 (log_perplexity)")
    plt.title("LDA模型困惑度曲线")

    plt.tight_layout()
    plt.savefig(output_chart, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"评估数据已保存到：{output_excel}")
    print(f"评估图表已保存到：{output_chart}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LDA困惑度与一致性评估工具")
    parser.add_argument("--input-file", default=None, help="分词文本路径（逐行文本）")
    parser.add_argument("--topic-min", type=int, default=2, help="最小主题数（默认: 2）")
    parser.add_argument("--topic-max", type=int, default=20, help="最大主题数（默认: 20）")
    parser.add_argument("--passes", type=int, default=5, help="训练 passes（默认: 5）")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数（默认: 4）")
    parser.add_argument("--output-excel", default=None, help="输出 Excel 文件路径（默认: lda_evaluation.xlsx）")
    parser.add_argument("--output-chart", default=None, help="输出图表文件路径（默认: lda_evaluation.png）")
    parser.add_argument("--font", default="Microsoft YaHei", help="字体名称（默认: Microsoft YaHei）")
    parser.add_argument("--font-path", default=None, help="字体文件路径（可选）")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="无头模式（默认: True）：不显示图形窗口，直接保存到文件",
    )
    parser.add_argument("--run-dir", default=None, help="运行工作目录（默认: runs/<timestamp_uuid>/）")
    return parser


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(args.input_file, "输入文本文件", allowed_suffixes={".txt"})
        output_excel = prepare_output_path(args.output_excel, "lda_evaluation.xlsx")
        output_chart = prepare_output_path(args.output_chart, "lda_evaluation.png")
        run_lda_evaluation(
            input_file,
            topic_min=int(args.topic_min),
            topic_max=int(args.topic_max),
            passes=int(args.passes),
            workers=int(args.workers),
            output_excel=output_excel,
            output_chart=output_chart,
            font=str(args.font),
            font_path=_resolve_font_path(args.font_path),
            headless=bool(args.headless),
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
