import argparse
import os
import re
import string
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import docx2txt
import jieba
import pandas as pd

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path

# ================= 全局变量及预编译对象 =================
CHINESE_PUNCTUATIONS = "，。！？【】（）％＃＠＆；：‘’“”、《》…「」"
ENGLISH_PUNCTUATIONS = string.punctuation.replace('"', '')
ALL_PUNCTUATIONS = ENGLISH_PUNCTUATIONS + CHINESE_PUNCTUATIONS
PUNCTUATION_PATTERN = re.compile("[" + re.escape(ALL_PUNCTUATIONS) + "]")

# ================= 函数定义 =================
def clean_text(text: str) -> str:
    """
    将文本中的所有英文和中文标点符号替换为空格，
    并将双引号（"）从文本中移除。
    """
    text = str(text)  # Convert to string if it's not already
    text = text.replace('"', '')
    return PUNCTUATION_PATTERN.sub(" ", text)

def segment_text_with_options(
    input_file,
    output_file,
    wordfreq_output_file=None,
    text_column='content',
    stopwords_file='stopwords.txt',
    userdict_file='userdict.txt',
    mode: str = 'search'  # 新增参数：支持 'precise'、'full' 或 'search'
):
    """
    对输入文件中的文本进行分词、预处理（清洗标点和双引号）、
    去除停用词，并统计词频。支持以下文件格式之一：
      - utf-8 编码的 txt 文件
      - Word 文档（.doc 或 .docx）
      - Excel 文件（.xls 或 .xlsx）
      - CSV 文件 (.csv)

    参数：
    - input_file: 输入文件路径（支持字符串或仅包含单一路径的列表/元组）
    - output_file: 分词结果的输出文件路径（txt 格式，制表符分隔）
    - wordfreq_output_file: 词频统计结果的输出路径；默认为 None，则自动生成文件名
    - text_column: 文本所在的列名（针对 Excel/CSV 文件有效，txt 和 Word 文件使用默认列名）
    - stopwords_file: 停用词文件路径；若为 None 或文件不存在，则不进行停用词过滤
    - userdict_file: 自定义词典文件路径；若为 None 或文件不存在，则跳过加载
    - mode: 分词模式，可选值：
         'precise' - 精确模式,
         'full'    - 全模式,
         'search'  - 搜索引擎模式（默认）
    """
    # 只允许输入单一文件路径
    if isinstance(input_file, (list, tuple)):
        if len(input_file) != 1:
            print("请只输入一个文件路径，然后退出程序。")
            sys.exit(1)
        input_file = input_file[0]

    # 根据文件扩展名读取文本
    ext = os.path.splitext(input_file)[1].lower()
    if ext == '.txt':
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"读取 txt 文件出错：{e}")
            sys.exit(1)
        df = pd.DataFrame({text_column: [text]})
    elif ext in ['.doc', '.docx']:
        try:
            text = docx2txt.process(input_file)
        except Exception as e:
            print(f"读取 Word 文档出错：{e}")
            sys.exit(1)
        df = pd.DataFrame({text_column: [text]})
    elif ext in ['.xls', '.xlsx']:
        try:
            df = pd.read_excel(input_file)
        except Exception as e:
            print(f"读取 Excel 文件出错：{e}")
            sys.exit(1)
    elif ext == '.csv':
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"读取 CSV 文件出错：{e}")
            sys.exit(1)
    else:
        print("不支持的文件格式！")
        sys.exit(1)

    # 预处理：填充空值，并清洗文本
    df[text_column] = df[text_column].fillna('').apply(clean_text)

    # 载入自定义词典（若提供且存在）
    if userdict_file and os.path.exists(userdict_file):
        jieba.load_userdict(userdict_file)
        print(f"已加载自定义词典：{userdict_file}")
    else:
        print("未提供或未找到自定义词典，跳过加载。")

    # 加载停用词表（若提供且存在），否则使用空集合
    if stopwords_file and os.path.exists(stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        print(f"已加载停用词文件：{stopwords_file}")
    else:
        if stopwords_file:
            print(f"未找到停用词文件 {stopwords_file}，将不进行停用词过滤。")
        else:
            print("未提供停用词文件，将不进行停用词过滤。")
        stopwords = set()

    def remove_stopwords(seg_text: str) -> str:
        """移除分词后的文本中的停用词，并返回处理后的字符串。"""
        return ' '.join(word for word in seg_text.split() if word not in stopwords)

    def jieba_segment(text: str) -> str:
        """根据 mode 对单个文本进行分词，返回以空格连接的词串。"""
        if mode == 'precise':
            seg = jieba.cut(text)
        elif mode == 'full':
            seg = jieba.cut(text, cut_all=True)
        elif mode == 'search':
            seg = jieba.cut_for_search(text)
        else:
            raise ValueError("未知的分词模式，请选择 'precise'、'full' 或 'search'")
        return ' '.join(seg)

    # 对文本进行分词，并去除停用词
    df['segmented'] = df[text_column].apply(jieba_segment)
    df['segmented_no_stopwords'] = df['segmented'].apply(remove_stopwords)

    # 统计词频（合并所有去除停用词后的分词结果）
    all_words = ' '.join(df['segmented_no_stopwords']).split()
    word_counts = Counter(all_words)

    # 输出词频最高的前 10 个词
    print("词频最高的前 10 个词：")
    for word, freq in word_counts.most_common(10):
        print(f"{word}: {freq}")

    # 若未指定词频输出文件路径，则自动生成
    if wordfreq_output_file is None:
        wordfreq_output_file = output_file.rsplit('.', 1)[0] + '_wordfreq.txt'
    try:
        with open(wordfreq_output_file, 'w', encoding='utf-8') as f:
            for word, freq in word_counts.most_common():
                f.write("{:<8}{:>2}\n".format(word, freq))
        print(f"词频统计结果已保存到 {wordfreq_output_file}")
    except Exception as e:
        print(f"写入词频统计文件出错：{e}")

    # 将分词结果输出为 txt 文件（使用制表符分隔）
    try:
        df.to_csv(output_file, sep='\t', index=False)
        print(f"分词完成！结果已保存到 {output_file}")
    except Exception as e:
        print(f"写入分词结果文件出错：{e}")


# ================= 示例调用 =================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="文档分词工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入文件路径（支持 txt/doc/docx/xls/xlsx/csv）",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="分词结果输出路径（默认: segmentation.txt）",
    )
    parser.add_argument(
        "--wordfreq-output-file",
        default=None,
        help="词频统计输出路径（可选）",
    )
    parser.add_argument(
        "--text-column",
        default="content",
        help="文本列名（Excel/CSV时有效，默认: content）",
    )
    parser.add_argument(
        "--stopwords-file",
        default=None,
        help="停用词表路径（可选）",
    )
    parser.add_argument(
        "--userdict-file",
        default=None,
        help="自定义词典路径（可选）",
    )
    parser.add_argument(
        "--mode",
        default="search",
        choices=["precise", "full", "search"],
        help="分词模式（默认: search）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(
    input_file: Path,
    output_file: Path,
    wordfreq_output_file: Optional[Path],
    text_column: str,
    stopwords_file: Optional[Path],
    userdict_file: Optional[Path],
    mode: str,
) -> None:
    segment_text_with_options(
        input_file=str(input_file),
        output_file=str(output_file),
        wordfreq_output_file=str(wordfreq_output_file) if wordfreq_output_file else None,
        text_column=text_column,
        stopwords_file=str(stopwords_file) if stopwords_file else None,
        userdict_file=str(userdict_file) if userdict_file else None,
        mode=mode,
    )


def cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入文件",
            allowed_suffixes={".txt", ".doc", ".docx", ".xls", ".xlsx", ".csv"},
        )
        output_file = prepare_output_path(args.output_file, "segmentation.txt")
        wordfreq_output_file = (
            prepare_output_path(args.wordfreq_output_file, "segmentation_wordfreq.txt")
            if args.wordfreq_output_file
            else None
        )
        stopwords_file = (
            require_input_path(
                args.stopwords_file,
                "停用词表",
                must_exist=True,
                allowed_suffixes={".txt"},
            )
            if args.stopwords_file
            else None
        )
        userdict_file = (
            require_input_path(
                args.userdict_file,
                "自定义词典",
                must_exist=True,
                allowed_suffixes={".txt"},
            )
            if args.userdict_file
            else None
        )

        main(
            input_file=input_file,
            output_file=output_file,
            wordfreq_output_file=wordfreq_output_file,
            text_column=args.text_column,
            stopwords_file=stopwords_file,
            userdict_file=userdict_file,
            mode=args.mode,
        )
    except Exception as exc:
        print(f"分词失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())


"""第一次使用需要安装第三方库:pip install jieba pandas docx2txt
支持以下文件格式之一：
      - utf-8 编码的 txt 文件
      - Word 文档（.doc 或 .docx）
      - Excel 文件（.xls 或 .xlsx）

    参数：
    - input_file: 输入文件路径（支持字符串或仅包含单一路径的列表/元组）
    - output_file: 分词结果的输出文件路径（txt 格式，制表符分隔）
    - wordfreq_output_file: 词频统计结果的输出路径；默认为 None，则自动生成文件名
    - text_column: 文本所在的列名（针对 Excel 文件有效，txt 和 Word 文件使用默认列名）
    - stopwords_file: 停用词文件路径；若为 None 或文件不存在，则不进行停用词过滤
    - userdict_file: 自定义词典文件路径；若为 None 或文件不存在，则跳过加载
    - mode: 分词模式，可选值：
         'precise' - 精确模式,
         'full'    - 全模式,
         'search'  - 搜索引擎模式（默认）
"""
