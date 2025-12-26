import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from snownlp import SnowNLP, sentiment

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from either CSV or Excel file."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        for encoding in ["utf-8", "gbk", "utf-8-sig"]:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except Exception:
                continue
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def main(
    model_path: Path,
    input_file: Path,
    output_excel: Path,
    output_chart1: Path,
    output_chart2: Path,
    histogram_bins: int = 50,
    chinese_font: str = "Microsoft YaHei",
    headless: bool = False,
) -> None:
    # 设置中文字体为指定的中文字体
    plt.rcParams['font.sans-serif'] = [chinese_font]  # 使用指定的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 1. 加载之前训练并保存的模型
    sentiment.load(str(model_path))

    # 2. 读取文件
    df = load_data(input_file)

    # 3. 计算情感得分
    df["情感得分"] = df["Q19"].apply(lambda x: SnowNLP(str(x)).sentiments)

    # 4. 创建Excel写入对象
    writer = pd.ExcelWriter(output_excel, engine='openpyxl')

    # 5. 将原始数据写入第一个工作表
    df.to_excel(writer, sheet_name='原始数据', index=False)

    # 6. 分区分析（10个区间）
    bins = [i/10 for i in range(11)]
    labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    df['情感得分区间'] = pd.cut(df['情感得分'], bins=bins, labels=labels, include_lowest=True)
    sentiment_dist = df['情感得分区间'].value_counts().sort_index()
    plot_data1 = pd.DataFrame({
        '情感得分区间': sentiment_dist.index,
        '评论数量': sentiment_dist.values
    })
    # 将分区数据写入第二个工作表
    plot_data1.to_excel(writer, sheet_name='分区统计(10区间)', index=False)

    # 7. 去掉“不分区数据(已排序)”的输出

    # 8. 计算不分区直方图的x和y轴数据
    hist_counts, hist_bins = np.histogram(df['情感得分'], bins=histogram_bins)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    plot_data3 = pd.DataFrame({
        '情感得分(直方图)': bin_centers,
        '评论数量': hist_counts
    })
    # 将直方图数据写入第四个工作表
    plot_data3.to_excel(writer, sheet_name='直方图数据', index=False)

    # 9. 保存Excel文件
    writer.close()

    # 10. 绘制分区柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(plot_data1['情感得分区间'], plot_data1['评论数量'], color='skyblue')
    plt.title('评论情感得分分布(10区间)', fontsize=15, fontweight='bold')
    plt.xlabel('情感得分区间', fontsize=12)
    plt.ylabel('评论数量', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(plot_data1['评论数量']):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_chart1, dpi=300, bbox_inches='tight')
    if not headless:
        plt.show()
    plt.close()

    # 11. 绘制不分区直方图
    plt.figure(figsize=(12, 6))
    plt.hist(df['情感得分'], bins=histogram_bins, color='skyblue', edgecolor='black')
    plt.title('评论情感得分分布(不分区)', fontsize=15, fontweight='bold')
    plt.xlabel('情感得分', fontsize=12)
    plt.ylabel('评论数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_chart2, dpi=300, bbox_inches='tight')
    if not headless:
        plt.show()
    plt.close()

    print(f"情感分析完成！结果已保存到: {output_excel}")
    print(f"分区图表已保存到: {output_chart1}")
    print(f"不分区图表已保存到: {output_chart2}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SnowNLP 情感分析工具")
    parser.add_argument(
        "--model-path",
        default=None,
        help="模型文件路径",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 csv 和 excel 文件路径",
    )
    parser.add_argument(
        "--output-excel",
        default=None,
        help="输出 Excel 文件路径（默认: comments_with_sentiment.xlsx）",
    )
    parser.add_argument(
        "--output-chart1",
        default=None,
        help="分区图表路径（默认: sentiment_distribution_10bins.png）",
    )
    parser.add_argument(
        "--output-chart2",
        default=None,
        help="不分区图表路径（默认: sentiment_distribution_continuous.png）",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=50,
        help="直方图 bins 数量（默认: 50）",
    )
    parser.add_argument(
        "--chinese-font",
        default="Microsoft YaHei",
        help="中文字体名称（默认: Microsoft YaHei）",
    )
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
            histogram_bins=args.histogram_bins,
            chinese_font=args.chinese_font,
            headless=args.headless,
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
