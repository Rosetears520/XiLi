# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
from gensim import corpora, models
from gensim.models import CoherenceModel
from tqdm import tqdm

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path

# 定义自定义 JSONEncoder 用于处理 NumPy 数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_texts(input_file: str) -> list:
    """读取并预处理文本数据（每行一篇文档）。"""
    processed_texts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="读取文件"):
            tokens = line.strip().split()
            if len(tokens) >= 2:  # 过滤空文档和过短文档
                processed_texts.append(tokens)
    logger.info(f"共读取 {len(processed_texts)} 篇文档。")
    return processed_texts


def build_corpus_and_dictionary(processed_texts: list, no_below: int, no_above: float, output_dir: str) -> tuple:
    """构建词典和语料库，并保存词典及语料统计信息。"""
    logger.info("构建词典和语料库...")
    dictionary = corpora.Dictionary(processed_texts)
    original_vocab_size = len(dictionary)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    filtered_vocab_size = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in tqdm(processed_texts, desc="构建语料库")]

    dictionary.save(os.path.join(output_dir, "dictionary.gensim"))
    corpus_stats = {
        "original_vocab_size": original_vocab_size,
        "filtered_vocab_size": filtered_vocab_size,
        "num_documents": len(corpus),
        "avg_doc_length": np.mean([len(doc) for doc in corpus]),
        "total_tokens": sum(len(doc) for doc in corpus)
    }
    with open(os.path.join(output_dir, "corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(corpus_stats, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    return dictionary, corpus


def train_lda_model(corpus: list, dictionary, num_topics: int, passes: int, alpha, eta,
                    random_seed: int, eval_every: int, chunksize: int, per_word_topics: bool,
                    minimum_probability: float, output_dir: str):
    """训练LDA模型并保存。"""
    logger.info("开始训练LDA模型...")
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha=alpha,
        eta=eta,
        random_state=random_seed,
        eval_every=eval_every,
        chunksize=chunksize,
        per_word_topics=per_word_topics,
        minimum_probability=minimum_probability
    )
    lda_model.save(os.path.join(output_dir, "lda_model.gensim"))
    return lda_model


def evaluate_model(lda_model, corpus: list, processed_texts: list, dictionary, num_topics: int,
                   coherence_measure: str, output_dir: str) -> dict:
    """评估LDA模型并保存评价指标。"""
    logger.info("评估模型质量...")
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=processed_texts,
        dictionary=dictionary,
        coherence=coherence_measure
    )
    coherence_score = coherence_model.get_coherence()

    # 利用所有主题词的并集计算主题多样性
    topic_words = lda_model.show_topics(num_topics=num_topics, num_words=20, formatted=False)
    all_topic_words = {word for topic in topic_words for word, _ in topic[1]}
    diversity_score = len(all_topic_words) / (num_topics * 20)

    evaluation = {
        "perplexity": perplexity,
        "coherence_score": coherence_score,
        "diversity_score": diversity_score,
        "topic_balance": list(lda_model.alpha) if hasattr(lda_model, 'alpha') else None,
        "word_distribution": list(lda_model.eta) if hasattr(lda_model, 'eta') else None
    }
    with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    return evaluation


def save_wordcloud_data(lda_model, dictionary, num_topics: int, output_dir: str):
    """保存每个主题的词频数据，供词云生成使用。"""
    logger.info("保存主题词频数据...")
    wordcloud_dir = os.path.join(output_dir, "wordcloud_data")
    os.makedirs(wordcloud_dir, exist_ok=True)
    for topic_id in range(num_topics):
        topic_terms = lda_model.get_topic_terms(topic_id, topn=50)
        with open(os.path.join(wordcloud_dir, f"topic_{topic_id}.txt"), "w", encoding="utf-8") as f:
            for word_id, prob in topic_terms:
                f.write(f"{dictionary[word_id]}\t{prob}\n")


def analyze_doc_topics(lda_model, corpus: list, minimum_probability: float, output_dir: str):
    """分析文档-主题分布，并保存结果。"""
    logger.info("分析文档-主题分布...")
    doc_topics = [lda_model.get_document_topics(doc, minimum_probability=minimum_probability)
                  for doc in tqdm(corpus, desc="分析文档主题")]
    with open(os.path.join(output_dir, "doc_topics.json"), "w", encoding="utf-8") as f:
        json.dump([[topic for topic in topics if topic[1] >= minimum_probability]
                   for topics in doc_topics], f, ensure_ascii=False, cls=NumpyEncoder)


def visualize_topics(lda_model, corpus: list, dictionary, output_dir: str) -> str:
    """生成并保存LDA可视化结果。"""
    logger.info("生成可视化结果...")
    vis_file = os.path.join(output_dir, "lda_visualization.html")
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, vis_file)
    return vis_file


def save_cluster_keywords_excel(lda_model, output_dir: str, num_topics: int, num_words: int = 10) -> str:
    """保存聚类编号-聚类关键词的Excel表格。"""
    logger.info("保存聚类关键词Excel表格...")
    topic_keywords = []
    for topic_id in range(num_topics):
        # 获取每个主题的关键词
        topic_terms = lda_model.show_topic(topic_id, topn=num_words)
        keywords = ", ".join([word for word, prob in topic_terms])
        topic_keywords.append({"Cluster ID": topic_id, "Keywords": keywords})
    df = pd.DataFrame(topic_keywords)
    excel_file = os.path.join(output_dir, "cluster_keywords.xlsx")
    df.to_excel(excel_file, index=False)
    return excel_file


def run_lda_analysis(
        input_file: str,
        output_dir: str = "output",
        num_topics: int = 5,
        no_below: int = 3,
        no_above: float = 0.5,
        passes: int = 20,
        alpha='auto',
        eta='auto',
        random_seed: int = 42,
        eval_every: int = 10,
        chunksize: int = 2000,
        per_word_topics: bool = True,
        minimum_probability: float = 0.01,
        coherence_measure: str = 'c_v'
) -> dict:
    """
    执行LDA主题建模分析

    参数:
        input_file: 输入文件路径（分词后的文本，每行一篇文档）
        output_dir: 输出目录（默认当前目录下的output文件夹）
        num_topics: 主题数量（默认5）
        no_below: 词频下限（默认3）
        no_above: 词频上限比例（默认0.5）
        passes: 训练迭代次数（默认20）
        alpha: 文档-主题分布的先验参数（默认'auto'）
        eta: 主题-词分布的先验参数（默认'auto'）
        random_seed: 随机种子（默认42）
        eval_every: 每隔多少次迭代评估模型（默认10）
        chunksize: 每次处理文档数（默认2000）
        per_word_topics: 是否计算每个词的主题分布（默认True）
        minimum_probability: 主题概率阈值（默认0.01）
        coherence_measure: 一致性评估方法（默认'c_v'）
    返回:
        包含模型、词典、语料、评价指标、可视化文件路径及聚类关键词Excel文件路径的字典
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    processed_texts = load_processed_texts(input_file)
    dictionary, corpus = build_corpus_and_dictionary(processed_texts, no_below, no_above, output_dir)
    lda_model = train_lda_model(corpus, dictionary, num_topics, passes, alpha, eta,
                                random_seed, eval_every, chunksize, per_word_topics,
                                minimum_probability, output_dir)
    evaluation = evaluate_model(lda_model, corpus, processed_texts, dictionary, num_topics, coherence_measure,
                                output_dir)
    save_wordcloud_data(lda_model, dictionary, num_topics, output_dir)
    analyze_doc_topics(lda_model, corpus, minimum_probability, output_dir)
    vis_file = visualize_topics(lda_model, corpus, dictionary, output_dir)
    # 保存聚类关键词Excel表格
    excel_file = save_cluster_keywords_excel(lda_model, output_dir, num_topics)

    logger.info(f"分析完成，总耗时 {(time.time() - start_time) / 60:.2f} 分钟")
    return {
        "model": lda_model,
        "dictionary": dictionary,
        "corpus": corpus,
        "evaluation": evaluation,
        "visualization_file": vis_file,
        "cluster_keywords_excel": excel_file
    }


def print_config(config: dict):
    """打印配置参数"""
    print("\n" + "=" * 50)
    print("LDA主题建模参数配置：")
    for key, value in config.items():
        print(f"{key:>15}: {value}")
    print("=" * 50 + "\n")


def print_results(results: dict, output_dir: str):
    """打印结果摘要"""
    print("\n分析结果摘要:")
    print(f"- 困惑度: {results['evaluation']['perplexity']:.3f}")
    print(f"- 一致性得分: {results['evaluation']['coherence_score']:.3f}")
    print(f"- 主题多样性: {results['evaluation']['diversity_score']:.3f}")
    print(f"\n词频数据已保存到: {os.path.join(output_dir, 'wordcloud_data')}")
    print(f"可视化文件: {results['visualization_file']}")
    print(f"聚类关键词Excel表格: {results['cluster_keywords_excel']}")

def _parse_alpha_eta(value: str):
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LDA主题建模分析工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入文件路径（分词文本）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认: outputs/lda）",
    )
    parser.add_argument("--num-topics", type=int, default=10, help="主题数量（默认: 10）")
    parser.add_argument("--no-below", type=int, default=3, help="词频下限（默认: 3）")
    parser.add_argument("--no-above", type=float, default=0.5, help="词频上限比例（默认: 0.5）")
    parser.add_argument("--passes", type=int, default=20, help="训练迭代次数（默认: 20）")
    parser.add_argument("--alpha", default="auto", help="文档-主题先验参数（默认: auto）")
    parser.add_argument("--eta", default="auto", help="主题-词先验参数（默认: auto）")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument("--eval-every", type=int, default=10, help="评估频率（默认: 10）")
    parser.add_argument("--chunksize", type=int, default=2000, help="每批文档数（默认: 2000）")
    parser.add_argument(
        "--per-word-topics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算词级主题分布（默认: True）",
    )
    parser.add_argument(
        "--minimum-probability",
        type=float,
        default=0.01,
        help="主题概率阈值（默认: 0.01）",
    )
    parser.add_argument(
        "--coherence-measure",
        default="c_v",
        help="一致性指标（默认: c_v）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(
    input_file: Path,
    output_dir: Path,
    num_topics: int,
    no_below: int,
    no_above: float,
    passes: int,
    alpha,
    eta,
    random_seed: int,
    eval_every: int,
    chunksize: int,
    per_word_topics: bool,
    minimum_probability: float,
    coherence_measure: str,
) -> None:
    config = {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "num_topics": num_topics,
        "no_below": no_below,
        "no_above": no_above,
        "passes": passes,
        "alpha": alpha,
        "eta": eta,
        "random_seed": random_seed,
        "eval_every": eval_every,
        "chunksize": chunksize,
        "per_word_topics": per_word_topics,
        "minimum_probability": minimum_probability,
        "coherence_measure": coherence_measure,
    }

    print_config(config)
    results = run_lda_analysis(**config)
    print_results(results, config["output_dir"])


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
        output_dir = prepare_output_path(args.output_dir, "outputs/lda")
        if output_dir.suffix:
            raise ValueError(f"输出目录应为文件夹路径：{output_dir}")
        if output_dir.exists() and output_dir.is_file():
            raise ValueError(f"输出目录不可为文件：{output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        main(
            input_file=input_file,
            output_dir=output_dir,
            num_topics=args.num_topics,
            no_below=args.no_below,
            no_above=args.no_above,
            passes=args.passes,
            alpha=_parse_alpha_eta(args.alpha),
            eta=_parse_alpha_eta(args.eta),
            random_seed=args.random_seed,
            eval_every=args.eval_every,
            chunksize=args.chunksize,
            per_word_topics=args.per_word_topics,
            minimum_probability=args.minimum_probability,
            coherence_measure=args.coherence_measure,
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
