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
    text = str(text)
    text = text.replace('"', '')
    return PUNCTUATION_PATTERN.sub(" ", text)

def segment_text_with_options(
    input_file,
    output_file,
    wordfreq_output_file=None,
    text_column='content',
    stopwords_file='stopwords.txt',
    userdict_file='userdict.txt',
    mode: str = 'search'
):
    """
    对输入文件中的文本进行分词、预处理、去除停用词，并统计词频。
    """
    if isinstance(input_file, (list, tuple)):
        if len(input_file) != 1:
            print("请只输入一个文件路径，然后退出程序。")
            sys.exit(1)
        input_file = input_file[0]

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

    df[text_column] = df[text_column].fillna('').apply(clean_text)

    if userdict_file and os.path.exists(userdict_file):
        jieba.load_userdict(userdict_file)
        print(f"已加载自定义词典：{userdict_file}")

    if stopwords_file and os.path.exists(stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        print(f"已加载停用词文件：{stopwords_file}")
    else:
        stopwords = set()

    def remove_stopwords(seg_text: str) -> str:
        return ' '.join(word for word in seg_text.split() if word not in stopwords)

    def jieba_segment(text: str) -> str:
        if mode == 'precise':
            seg = jieba.cut(text)
        elif mode == 'full':
            seg = jieba.cut(text, cut_all=True)
        elif mode == 'search':
            seg = jieba.cut_for_search(text)
        else:
            raise ValueError("未知的分词模式")
        return ' '.join(seg)

    df['segmented'] = df[text_column].apply(jieba_segment)
    df['segmented_no_stopwords'] = df['segmented'].apply(remove_stopwords)

    all_words = ' '.join(df['segmented_no_stopwords']).split()
    word_counts = Counter(all_words)

    if wordfreq_output_file is None:
        wordfreq_output_file = output_file.rsplit('.', 1)[0] + '_wordfreq.txt'
    try:
        with open(wordfreq_output_file, 'w', encoding='utf-8') as f:
            for word, freq in word_counts.most_common():
                f.write("{:<8}{:>2}\n".format(word, freq))
    except Exception as e:
        print(f"写入词频统计文件出错：{e}")

    try:
        df.to_csv(output_file, sep='\t', index=False)
        print(f"分词完成！结果已保存到 {output_file}")
    except Exception as e:
        print(f"写入分词结果文件出错：{e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="文档分词工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入文件路径",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="分词结果输出路径",
    )
    parser.add_argument(
        "--wordfreq-output-file",
        default=None,
        help="词频统计输出路径",
    )
    parser.add_argument(
        "--text-column",
        default="content",
        help="文本列名",
    )
    parser.add_argument(
        "--stopwords-file",
        default=None,
        help="停用词表路径",
    )
    parser.add_argument(
        "--userdict-file",
        default=None,
        help="自定义词典路径",
    )
    parser.add_argument(
        "--mode",
        default="search",
        choices=["precise", "full", "search"],
        help="分词模式",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录",
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
            allowed_suffixes={'.txt', '.doc', '.docx', '.xls', '.xlsx', '.csv'},
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
                allowed_suffixes={'.txt'},
            )
            if args.stopwords_file
            else None
        )
        userdict_file = (
            require_input_path(
                args.userdict_file,
                "自定义词典",
                must_exist=True,
                allowed_suffixes={'.txt'},
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
