# Fixtures (minimal Excel workbooks)

This folder contains tiny `.xlsx` fixtures for regression/acceptance checks across:
- Entropy-weight TOPSIS branching (`group_cols` / `year_col` / same-name dedupe / multi group cols)
- RankTest unified interface (`mode=questions|group`, including multi-column `group_cols`)

## Entropy TOPSIS

### `entropy_topsis/entropy_topsis_group_only_multi.xlsx`

- Columns: `省份`, `行业`, `指标1`, `指标2`
- Intended branch: group-only (`group_cols=省份,行业`, `year_col=` empty)
- Multi group cols: yes (`省份` + `行业`)

### `entropy_topsis/entropy_topsis_group_year_multi.xlsx`

- Columns: `年份`, `省份`, `行业`, `指标1`, `指标2`
- Intended branches:
  - year-only (`year_col=年份`, `group_cols=` empty)
  - group+year (`year_col=年份`, `group_cols=省份,行业`)
  - same-name dedupe (e.g. `year_col=年份`, `group_cols=年份,省份,行业` → dedupe to `省份,行业`)

## RankTest

### `ranktest/ranktest_questions.xlsx`

- Columns: `Q1`, `Q2`, `BIU`
- Intended mode: `mode=questions`
  - `Q1` has 2 groups → Mann-Whitney U
  - `Q2` has 3 groups → Kruskal-Wallis

### `ranktest/ranktest_group_multi.xlsx`

- Columns: `区域`, `性别`, `满意度`, `时长`
- Intended mode: `mode=group`
- Multi group cols: yes (`group_cols=区域,性别`)

## Suggested CLI usage (post-refactor target)

Entropy TOPSIS (multi `group_cols`):
`python -m xili.cli topsis --input-file tests/fixtures/entropy_topsis/entropy_topsis_group_year_multi.xlsx --year-col 年份 --group-cols 省份,行业`

Same-name dedupe:
`python -m xili.cli topsis --input-file tests/fixtures/entropy_topsis/entropy_topsis_group_year_multi.xlsx --year-col 年份 --group-cols 年份,省份,行业`

RankTest:
- questions: `python -m xili.cli ranktest --mode questions --input-file tests/fixtures/ranktest/ranktest_questions.xlsx --question-cols Q1,Q2 --dv-col BIU`
- group: `python -m xili.cli ranktest --mode group --input-file tests/fixtures/ranktest/ranktest_group_multi.xlsx --group-cols 区域,性别`

