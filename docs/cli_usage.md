# CLI Usage Guide

This guide describes how to use the command-line interface (CLI) for the data analysis tools.

## Unified CLI (Recommended)

The `main.py` script provides a unified entry point for all tools, including core analysis, visualization, and preprocessing.

### General Syntax
```bash
uv run -- python main.py <command> [options]
```
If no command is provided, it defaults to launching the GUI.

### Core Analysis Tools

#### Entropy TOPSIS
```bash
uv run -- python main.py topsis --input-file data/input.xlsx --group-cols Province --year-col Year --id-cols Code --negative-indicators Metric1,Metric2 --output-file topsis_results.xlsx
```

#### Rank Sum Test
```bash
uv run -- python main.py ranktest --mode questions --input-file data/input.xlsx --question-cols Q1,Q2,Q3 --dv-col BIU
```

### Visualization and NLP

#### Word Cloud
```bash
uv run -- python main.py wordcloud --input-file data/word_freq.txt --output-image wordcloud.png
```

#### LDA Topic Modeling
```bash
uv run -- python main.py lda_model --input-file data/segmented.txt --output-dir outputs/lda --num-topics 10
uv run -- python main.py lda_eval --input-file data/segmented.txt --output-excel lda_evaluation.xlsx --topic-min 2 --topic-max 20
```

#### Sentiment Analysis (SnowNLP)
```bash
uv run -- python main.py snowlp_train --negative-file data/negative.txt --positive-file data/positive.txt --output-model sentiment.marshal
uv run -- python main.py snowlp_analysis --model-path data/sentiment.marshal --input-file data/input.xlsx --output-excel comments_with_sentiment.xlsx
```

#### Image Processing
```bash
uv run -- python main.py binary_image --input-file data/input.png --output-file binary.png --threshold 128
```

### Data Preprocessing and Clustering

#### K-means Clustering
```bash
uv run -- python main.py kmeans --input-file data/kmeans.xlsx --output-dir outputs/kmeans --max-k 10
```

#### Average Growth Rate
```bash
uv run -- python main.py growth_rate --input-file data/input.csv --output-file average_growth_rate_result.csv
```

#### Yearly Averaging
```bash
uv run -- python main.py yearly_average --input-file data/input.xlsx --output-file yearly_averages.xlsx --cols T,O,E --rename T=T_avg --rename O=O_avg
```

## Legacy Commands

Direct script execution is still supported but discouraged in favor of the unified `main.py`.

```bash
uv run -- python word_cloud.py --help
uv run -- python kmeans.py --help
```