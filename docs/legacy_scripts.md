# Legacy Scripts

The following legacy scripts are maintained for backward compatibility. New users should use the unified entry point `main.py` or the English-named equivalents.

## Entropy TOPSIS Legacy Scripts (in `Legacy Scripts/`)

- **Province Grouping & Year**:
  `uv run -- python "熵权topsis法 - 以省份为分组 -年份.py" --input-file data/input.xlsx --output-file topsis_by_province_year.xlsx`
- **With Province Column**:
  `uv run -- python "熵权topsis法---加入省份列.py" --input-file data/input.xlsx --output-file topsis_with_region.xlsx`
- **Single Province**:
  `uv run -- python "熵权topsis法-单省份.py" --input-file data/input.xlsx --output-file topsis_single.xlsx --name-col SchoolName`
- **By Year**:
  `uv run -- python "熵权topsis法-年份.py" --input-file data/input.xlsx --output-file topsis_by_year.xlsx`

## Rank Sum Test Legacy Scripts (in `Legacy Scripts/`)

- **Standard Rank Test**:
  `uv run -- python "秩和检验.py" --input-file data/input.xlsx --output-file rank_test_summary.xlsx --question-cols Q1,Q2,Q3,Q4,Q5 --dv-col BIU`
- **Grouped Rank Test**:
  `uv run -- python "秩和检验---分组.py" --input-file data/input.xlsx --output-file group_rank_test_results.xlsx --group-col Region`

## Redirected Root Scripts

The following scripts in the root directory have been renamed to English. The Chinese versions are kept for compatibility but call the new scripts or subcommands.

| Old Chinese Filename | New English Filename | Unified Command |
| :--- | :--- | :--- |
| `词云图.py` | `word_cloud.py` | `python main.py wordcloud` |
| `黑白图片.py` | `binary_image.py` | `python main.py binary_image` |
| `计算年份平均值.py` | `yearly_average.py` | `python main.py yearly_average` |
| `平均增长率.py` | `average_growth_rate.py` | `python main.py growth_rate` |
| `K-means.py` | `kmeans.py` | `python main.py kmeans` |
| `LDA困惑度和一致性.py` | `lda_evaluation.py` | `python main.py lda_eval` |
| `LDA模型.py` | `lda_model.py` | `python main.py lda_model` |
| `Snowlp训练模型.py` | `snownlp_train.py` | `python main.py snowlp_train` |
| `SnowNLP情感分析.py` | `snownlp_sentiment.py` | `python main.py snowlp_analysis` |