from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from snownlp import SnowNLP, sentiment

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def load_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        for encoding in ["utf-8", "gbk", "utf-8-sig"]:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except Exception:
                continue
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def _detect_text_col(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for name in ["Q19", "content", "评论", "comment", "text", "文本"]:
        if name in df.columns:
            return name
    # fallback: first object column
    object_cols = [c for c in df.columns if df[c].dtype == object]
    if object_cols:
        return object_cols[0]
    return str(df.columns[0])


def _normalize_snownlp_model_path_for_load(model_path: Path) -> Path:
    """
    SnowNLP 在 Python3 下会在 load/save 时自动追加 ".3" 后缀。
    因此如果用户上传/选择了实际存在的 "sentiment.marshal.3" 文件，这里需要剥离一次 ".3"，
    避免 SnowNLP 再追加导致尝试读取 ".3.3"。
    """

    if model_path.suffix == ".3":
        return Path(str(model_path)[: -len(".3")])
    return model_path


def main(
    model_path: Path,
    input_file: Path,
    output_excel: Path,
    output_chart1: Path,
    output_chart2: Path,
    *,
    text_col: Optional[str],
    histogram_bins: int = 50,
    chinese_font: str = "Microsoft YaHei",
    headless: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    if headless:
        plt.switch_backend("Agg")

    plt.rcParams["font.sans-serif"] = [chinese_font]
    plt.rcParams["axes.unicode_minus"] = False

    model_path_for_load = _normalize_snownlp_model_path_for_load(model_path)
    try:
        sentiment.load(str(model_path_for_load))
    except Exception as exc:
        raise RuntimeError(
            "加载模型失败："
            f"{model_path}（提示：Python3 下通常是 sentiment.marshal.3；传入 sentiment.marshal 也可以）: {exc}"
        ) from exc
    df = load_data(input_file)
    text_col_final = _detect_text_col(df, text_col)
    if text_col_final not in df.columns:
        raise ValueError(f"文本列不存在：{text_col_final}")

    df["情感得分"] = df[text_col_final].apply(lambda x: SnowNLP(str(x)).sentiments)

    writer = pd.ExcelWriter(output_excel, engine="openpyxl")
    df.to_excel(writer, sheet_name="原始数据", index=False)

    bins = [i / 10 for i in range(11)]
    labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    df["情感得分区间"] = pd.cut(df["情感得分"], bins=bins, labels=labels, include_lowest=True)
    sentiment_dist = df["情感得分区间"].value_counts().sort_index()
    plot_data1 = pd.DataFrame({"情感得分区间": sentiment_dist.index, "评论数量": sentiment_dist.values})
    plot_data1.to_excel(writer, sheet_name="分区统计(10区间)", index=False)

    hist_counts, hist_bins = np.histogram(df["情感得分"], bins=histogram_bins)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    plot_data3 = pd.DataFrame({"情感得分(直方图)": bin_centers, "评论数量": hist_counts})
    plot_data3.to_excel(writer, sheet_name="直方图数据", index=False)
    writer.close()

    plt.figure(figsize=(12, 6))
    plt.bar(plot_data1["情感得分区间"], plot_data1["评论数量"], color="skyblue")
    plt.title("评论情感得分分布(10区间)", fontsize=15, fontweight="bold")
    plt.xlabel("情感得分区间", fontsize=12)
    plt.ylabel("评论数量", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_chart1, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(df["情感得分"], bins=histogram_bins, color="skyblue", edgecolor="black")
    plt.title("评论情感得分分布(不分区)", fontsize=15, fontweight="bold")
    plt.xlabel("情感得分", fontsize=12)
    plt.ylabel("评论数量", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_chart2, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"情感分析完成！结果已保存到: {output_excel}")
    print(f"文本列: {text_col_final}")
    print(f"分区图表已保存到: {output_chart1}")
    print(f"不分区图表已保存到: {output_chart2}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SnowNLP 情感分析工具")
    parser.add_argument("--model-path", default=None, help="模型文件路径")
    parser.add_argument("--input-file", default=None, help="输入 csv 和 excel 文件路径")
    parser.add_argument("--text-col", default=None, help="文本列名（可选；默认自动识别，优先 Q19）")
    parser.add_argument("--output-excel", default=None, help="输出 Excel 文件路径（默认: comments_with_sentiment.xlsx）")
    parser.add_argument("--output-chart1", default=None, help="分区图表路径（默认: sentiment_distribution_10bins.png）")
    parser.add_argument(
        "--output-chart2",
        default=None,
        help="不分区图表路径（默认: sentiment_distribution_continuous.png）",
    )
    parser.add_argument("--histogram-bins", type=int, default=50, help="直方图 bins 数量（默认: 50）")
    parser.add_argument("--chinese-font", default="Microsoft YaHei", help="中文字体名称（默认: Microsoft YaHei）")
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
        model_path = require_input_path(args.model_path, "模型文件")
        input_file = require_input_path(
            args.input_file,
            "输入 csv 和 excel 文件",
            allowed_suffixes={".xlsx", ".xls", ".csv"},
        )
        output_excel = prepare_output_path(args.output_excel, "comments_with_sentiment.xlsx")
        output_chart1 = prepare_output_path(args.output_chart1, "sentiment_distribution_10bins.png")
        output_chart2 = prepare_output_path(args.output_chart2, "sentiment_distribution_continuous.png")

        main(
            model_path=model_path,
            input_file=input_file,
            output_excel=output_excel,
            output_chart1=output_chart1,
            output_chart2=output_chart2,
            text_col=args.text_col,
            histogram_bins=int(args.histogram_bins),
            chinese_font=str(args.chinese_font),
            headless=bool(args.headless),
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
