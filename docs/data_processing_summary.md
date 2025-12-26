# Data Processing Summary Document

> This document serves as a project-level summary template for the data analysis toolbox, recording the environment, inputs, processing steps, results, and reproducibility information for each run.

---

## 1. Environment

| Item | Version/Config |
|------|----------|
| Python Version | 3.12.8 |
| Package Manager | uv + .venv |
| Execution Command | `uv run -- python main.py <command> [args...]` |

---

## 2. Input Requirements

### 2.1 Core Analysis Tools

| Tool | Input Format | Required Columns/Params | Description |
|----------|----------|-------------|------|
| Entropy TOPSIS | csv 和 excel | Data columns, Optional: group_cols, year_col, id_cols, negative_indicators | Supports multi-column and yearly grouping modes |
| Rank Sum Test | csv 和 excel | Data columns, question_cols or group_cols, dv_col | Supports 'questions' and 'group' modes |

### 2.2 Descriptive Statistics & Imputation

| Tool | Input Format | Required Columns/Params | Description |
|----------|----------|-------------|------|
| Yearly Average | csv 和 excel | Column list (cols), Optional: rename pairs | Calculates average values per stock code or ID |
| Growth Rate | csv 和 excel | Data columns, Province column | Fills missing values using average growth rates |

### 2.3 Clustering & NLP

| Tool | Input Format | Required Columns/Params | Description |
|----------|----------|-------------|------|
| K-means | csv 和 excel | Numerical columns | Automatic optimal K determination and visualization |
| SnowNLP Analysis | csv 和 excel + Model | Model path, Data columns | Sentiment scoring and distribution visualization |
| LDA Model | Text (.txt) | Segmented text (one doc per line) | Topic extraction and visualization |
| LDA Evaluation | Text (.txt) | Segmented text, Topic range | Perplexity and coherence evaluation |
| SnowNLP Training | Text (.txt) | Positive/Negative corpora | Train custom sentiment models |

---

## 3. Processing Steps

### 3.1 Entropy TOPSIS
1. Load CSV/Excel data.
2. Preprocess (handle missing/outlier values).
3. Normalize (apply `eps_shift` non-negative translation).
4. Calculate entropy weights.
5. Compute closeness using TOPSIS method.
6. (Optional) Group-wise calculation.
7. Save results to Excel.

### 3.2 Rank Sum Test
1. Load CSV/Excel data.
2. Group data based on mode (questions/group).
3. Execute non-parametric tests (Mann-Whitney U or Kruskal-Wallis H).
4. Calculate statistics and p-values.
5. Determine significance based on alpha.
6. Save results to Excel.

---

## 4. Output Summary

| Tool | Output Type | Default Filename/Path |
|----------|--------------|-----------------|
| Entropy TOPSIS | Excel (.xlsx) | `entropy_topsis.xlsx` |
| Rank Sum Test | Excel (.xlsx) | `rank_tests.xlsx` |
| Yearly Average | csv 和 excel | `yearly_averages.xlsx` |
| Growth Rate | csv 和 excel | `average_growth_rate_result.csv` |
| K-means | Excel + Images | `outputs/kmeans/` |
| SnowNLP Analysis | Excel + Images | `comments_with_sentiment.xlsx` |

---

## 5. Data Policy

- Raw CSV/Excel datasets should not be committed to the repository.
- Non-sensitive samples should reside in `tests/fixtures/`.
- Run artifacts in `runs/` are ignored by git.
- Ensure sensitive data is de-identified before processing or sharing.
