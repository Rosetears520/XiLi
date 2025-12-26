import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import gensim
import jieba
import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore
from matplotlib import font_manager  # 字体管理模块

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path

def resolve_font_path(font_input: Optional[str]) -> Optional[str]:
    """
    Resolve font path from input (either a path or a font family name).
    """
    if not font_input:
        return None
    
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

def load_texts(file_path):
    """
    从txt文件中加载文本，每行视为一个文档，并使用jieba进行分词
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = jieba.lcut(line)
                texts.append(tokens)
    return texts

def run_lda_evaluation(file_path, topic_min=2, topic_max=20, passes=10, workers=4,
                       output_excel='lda_evaluation.xlsx', output_chart='lda_evaluation.png',
                       font='Microsoft YaHei', font_path=None, headless=False):
    """
    根据指定的主题数范围训练LDA模型，并计算每个模型的一致性和困惑度，
    绘制曲线图并将结果数据保存到Excel文件中。

    参数:
      - file_path: 文本文件的路径
      - topic_min: 最小主题数
      - topic_max: 最大主题数
      - passes: LDA模型训练时passes次数
      - workers: 使用LdaMulticore时的线程数（自定义并行数）
      - output_excel: 保存评估数据的Excel文件名
      - output_chart: 保存曲线图的图片文件名
      - font: 字体名称，默认为"Microsoft YaHei"
      - font_path: 可选的字体文件路径，如传入则优先使用此字体
      - headless: 是否以无头模式运行（不显示图形窗口，直接保存到文件）
    """
    # 1. 加载并预处理文本
    texts = load_texts(file_path)

    # 2. 构建词典和语料库
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 3. 定义主题数范围
    topic_range = range(topic_min, topic_max + 1)

    coherence_values = []
    perplexity_values = []
    model_list = []

    # 4. 分别训练不同主题数的LDA模型并计算评价指标
    for num_topics in topic_range:
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            per_word_topics=True,
            workers=workers
        )
        model_list.append(model)

        # 计算一致性得分（采用 c_v 作为评价指标）
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        coherence_values.append(coherence)

        # 计算困惑度（log_perplexity值）
        perplexity = model.log_perplexity(corpus)
        perplexity_values.append(perplexity)

    # 5. 设置中文字体，根据传入的font或font_path参数
    if font_path is not None:
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        plt.rcParams['font.sans-serif'] = [font]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 6. 绘制一致性和困惑度曲线
    plt.figure(figsize=(12, 6))

    # 绘制一致性曲线
    plt.subplot(1, 2, 1)
    plt.plot(list(topic_range), coherence_values, marker='o')
    plt.xlabel("主题数")
    plt.ylabel("一致性得分")
    plt.title("LDA模型一致性曲线")

    # 绘制困惑度曲线
    plt.subplot(1, 2, 2)
    plt.plot(list(topic_range), perplexity_values, marker='o')
    plt.xlabel("主题数")
    plt.ylabel("困惑度 (log_perplexity)")
    plt.title("LDA模型困惑度曲线")

    plt.tight_layout()

    # 根据headless参数决定是否显示图形窗口
    if headless:
        plt.savefig(output_chart, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"评估图表已保存到：{output_chart}")
    else:
        plt.show()

    # 7. 将绘图数据保存到Excel文件
    data = {
        '主题数': list(topic_range),
        '一致性得分': coherence_values,
        '困惑度 (log_perplexity)': perplexity_values
    }
    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False)
    print(f"评估数据已保存到Excel文件：{output_excel}")

# -------------------------------
# 示例调用：请根据实际情况修改以下参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LDA困惑度与一致性评估工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="分词文本路径（逐行文本）",
    )
    parser.add_argument("--topic-min", type=int, default=2, help="最小主题数（默认: 2）")
    parser.add_argument("--topic-max", type=int, default=20, help="最大主题数（默认: 20）")
    parser.add_argument("--passes", type=int, default=5, help="训练 passes（默认: 5）")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数（默认: 4）")
    parser.add_argument(
        "--output-excel",
        default=None,
        help="输出 Excel 文件路径（默认: lda_evaluation.xlsx）",
    )
    parser.add_argument(
        "--output-chart",
        default=None,
        help="输出图表文件路径（默认: lda_evaluation.png）",
    )
    parser.add_argument("--font", default="Microsoft YaHei", help="字体名称（默认: Microsoft YaHei）")
    parser.add_argument("--font-path", default=None, help="字体文件路径（可选）")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式：不显示图形窗口，直接保存到文件",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(
    input_file: Path,
    output_excel: Path,
    output_chart: Path,
    topic_min: int,
    topic_max: int,
    passes: int,
    workers: int,
    font: str,
    font_path: Optional[Path],
    headless: bool,
) -> None:
    run_lda_evaluation(
        file_path=str(input_file),
        topic_min=topic_min,
        topic_max=topic_max,
        passes=passes,
        workers=workers,
        output_excel=str(output_excel),
        output_chart=str(output_chart),
        font=font,
        font_path=str(font_path) if font_path else None,
        headless=headless,
    )


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入文本文件",
            allowed_suffixes={".txt"},
        )
        output_excel = prepare_output_path(args.output_excel, "lda_evaluation.xlsx")
        output_chart = prepare_output_path(args.output_chart, "lda_evaluation.png")
        font_path = resolve_font_path(args.font_path)

        main(
            input_file=input_file,
            output_excel=output_excel,
            output_chart=output_chart,
            topic_min=args.topic_min,
            topic_max=args.topic_max,
            passes=args.passes,
            workers=args.workers,
            font=args.font,
            font_path=font_path,
            headless=args.headless,
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
