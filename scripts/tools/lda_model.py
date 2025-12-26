from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyLDAvis.gensim_models
from gensim import corpora, models
from gensim.models import CoherenceModel
from tqdm import tqdm

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: ANN001
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _parse_alpha_eta(value: str):
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_processed_texts(input_file: Path) -> list[list[str]]:
    processed_texts: list[list[str]] = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="读取文件"):
            tokens = line.strip().split()
            if len(tokens) >= 2:
                processed_texts.append(tokens)
    if not processed_texts:
        raise ValueError("输入文本为空或过短，无法训练 LDA。")
    return processed_texts


def build_corpus_and_dictionary(
    processed_texts: list[list[str]],
    *,
    no_below: int,
    no_above: float,
    output_dir: Path,
) -> tuple[corpora.Dictionary, list[list[tuple[int, int]]]]:
    dictionary = corpora.Dictionary(processed_texts)
    original_vocab_size = len(dictionary)
    dictionary.filter_extremes(no_below=int(no_below), no_above=float(no_above))
    filtered_vocab_size = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in tqdm(processed_texts, desc="构建语料库")]

    dictionary.save(str(output_dir / "dictionary.gensim"))
    corpus_stats = {
        "original_vocab_size": original_vocab_size,
        "filtered_vocab_size": filtered_vocab_size,
        "num_documents": len(corpus),
        "avg_doc_length": float(np.mean([len(doc) for doc in corpus])) if corpus else 0.0,
        "total_tokens": int(sum(len(doc) for doc in corpus)),
    }
    (output_dir / "corpus_stats.json").write_text(
        json.dumps(corpus_stats, indent=2, ensure_ascii=False, cls=_NumpyEncoder),
        encoding="utf-8",
    )
    return dictionary, corpus


def train_lda_model(
    corpus: list[list[tuple[int, int]]],
    dictionary: corpora.Dictionary,
    *,
    num_topics: int,
    passes: int,
    alpha,
    eta,
    eval_every: int,
    chunksize: int,
    per_word_topics: bool,
    minimum_probability: float,
    output_dir: Path,
):
    # Do NOT set any random seed / random_state here (per request).
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=int(num_topics),
        passes=int(passes),
        alpha=alpha,
        eta=eta,
        eval_every=int(eval_every),
        chunksize=int(chunksize),
        per_word_topics=bool(per_word_topics),
        minimum_probability=float(minimum_probability),
    )
    lda_model.save(str(output_dir / "lda_model.gensim"))
    return lda_model


def evaluate_model(
    lda_model,
    corpus: list[list[tuple[int, int]]],
    processed_texts: list[list[str]],
    dictionary: corpora.Dictionary,
    *,
    num_topics: int,
    coherence_measure: str,
    output_dir: Path,
) -> dict:
    perplexity = float(lda_model.log_perplexity(corpus))
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=processed_texts,
        dictionary=dictionary,
        coherence=str(coherence_measure),
    )
    coherence_score = float(coherence_model.get_coherence())

    topic_words = lda_model.show_topics(num_topics=int(num_topics), num_words=20, formatted=False)
    all_topic_words = {word for topic in topic_words for word, _ in topic[1]}
    diversity_score = float(len(all_topic_words) / max(1, int(num_topics) * 20))

    evaluation = {
        "perplexity": perplexity,
        "coherence_score": coherence_score,
        "diversity_score": diversity_score,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False, cls=_NumpyEncoder),
        encoding="utf-8",
    )
    return evaluation


def save_wordcloud_data(lda_model, *, num_topics: int, output_dir: Path) -> None:
    wordcloud_dir = output_dir / "wordcloud_data"
    wordcloud_dir.mkdir(parents=True, exist_ok=True)
    for topic_id in range(int(num_topics)):
        topic_terms = lda_model.show_topic(topic_id, topn=200)
        freq_map = {word: float(prob) for word, prob in topic_terms}
        out_path = wordcloud_dir / f"topic_{topic_id}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for word, prob in sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True):
                f.write(f"{word} {prob}\n")


def save_cluster_keywords_excel(lda_model, *, num_topics: int, output_dir: Path, num_words: int = 20) -> Path:
    rows: list[dict[str, object]] = []
    for topic_id in range(int(num_topics)):
        topic_terms = lda_model.show_topic(topic_id, topn=int(num_words))
        keywords = ", ".join([word for word, _ in topic_terms])
        rows.append({"Cluster ID": topic_id, "Keywords": keywords})
    df = pd.DataFrame(rows)
    excel_file = output_dir / "cluster_keywords.xlsx"
    df.to_excel(excel_file, index=False)
    return excel_file


def visualize_topics(lda_model, corpus, dictionary, *, output_dir: Path) -> Path:
    import pyLDAvis

    vis_file = output_dir / "lda_vis.html"
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, str(vis_file))
    return vis_file


def write_summary(
    *,
    output_dir: Path,
    evaluation: dict,
    cluster_keywords_excel: Path,
    visualization_file: Path,
    elapsed_seconds: float,
) -> None:
    lines = [
        "# LDA 模型结果摘要",
        "",
        f"- 困惑度: {evaluation.get('perplexity')}",
        f"- 一致性得分: {evaluation.get('coherence_score')}",
        f"- 主题多样性: {evaluation.get('diversity_score')}",
        f"- 主题关键词Excel: {cluster_keywords_excel.name}",
        f"- 可视化HTML: {visualization_file.name}",
        f"- 耗时(秒): {elapsed_seconds:.2f}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LDA主题建模分析工具")
    parser.add_argument("--input-file", default=None, help="输入文件路径（分词文本，每行一篇文档）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认: outputs/lda）")
    parser.add_argument("--num-topics", type=int, default=10, help="主题数量（默认: 10）")
    parser.add_argument("--no-below", type=int, default=3, help="去除低频词（默认: 3）")
    parser.add_argument("--no-above", type=float, default=0.5, help="去除高频词比例（默认: 0.5）")
    parser.add_argument("--passes", type=int, default=20, help="训练轮数（默认: 20）")
    parser.add_argument("--alpha", default="auto", help="文档-主题先验分布（默认: auto）")
    parser.add_argument("--eta", default="auto", help="主题-词先验分布（默认: auto）")
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
        help="最小概率阈值（默认: 0.01）",
    )
    parser.add_argument("--coherence-measure", default="c_v", help="一致性评分标准（默认: c_v）")
    parser.add_argument("--run-dir", default=None, help="运行工作目录（默认: runs/<timestamp_uuid>/）")
    return parser


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(args.input_file, "输入文本文件", allowed_suffixes={".txt"})
        output_dir = prepare_output_path(args.output_dir, "outputs/lda")
        if output_dir.suffix:
            raise ValueError(f"输出目录应为文件夹路径：{output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        processed_texts = load_processed_texts(input_file)
        dictionary, corpus = build_corpus_and_dictionary(
            processed_texts,
            no_below=args.no_below,
            no_above=args.no_above,
            output_dir=output_dir,
        )
        lda_model = train_lda_model(
            corpus,
            dictionary,
            num_topics=args.num_topics,
            passes=args.passes,
            alpha=_parse_alpha_eta(args.alpha),
            eta=_parse_alpha_eta(args.eta),
            eval_every=args.eval_every,
            chunksize=args.chunksize,
            per_word_topics=args.per_word_topics,
            minimum_probability=args.minimum_probability,
            output_dir=output_dir,
        )
        evaluation = evaluate_model(
            lda_model,
            corpus,
            processed_texts,
            dictionary,
            num_topics=args.num_topics,
            coherence_measure=args.coherence_measure,
            output_dir=output_dir,
        )
        save_wordcloud_data(lda_model, num_topics=args.num_topics, output_dir=output_dir)
        visualization_file = visualize_topics(lda_model, corpus, dictionary, output_dir=output_dir)
        excel_file = save_cluster_keywords_excel(lda_model, num_topics=args.num_topics, output_dir=output_dir)
        write_summary(
            output_dir=output_dir,
            evaluation=evaluation,
            cluster_keywords_excel=excel_file,
            visualization_file=visualization_file,
            elapsed_seconds=time.time() - start,
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
