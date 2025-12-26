# Legacy Scripts Baseline (Merge Acceptance)

This document freezes the current CLI contract of 6 legacy scripts (args/defaults/output filenames/sheets/key columns),
so we can refactor to a unified CLI while still validating backward compatibility.

---

# 旧脚本基线（合并验收基准）

目的：冻结 6 个旧脚本的「参数/默认值/输出文件名/Sheet 名/关键列」，用于后续统一 CLI / wrapper 合并时的对齐验收。

## 通用约定（6 个脚本一致）

- `--run-dir`：不传时会创建并进入 `runs/<YYYYMMDD-HHMMSS>_<6hex>/`，默认输出文件写入该目录。
- `--input-file`：相对路径先在当前工作目录（run dir）查找；不存在则按项目根目录解析。
- Excel 读取：均为 `pd.read_excel(input_file)`（默认读取第一个工作表）。

## 熵权 TOPSIS（4 个脚本）

共同点：
- `--negative-indicators`：默认留空时，使用 `xili.entropy_topsis.DEFAULT_NEGATIVE_INDICATORS`：
  - `服务业二氧化硫排放量/服务业增加值 二氧化硫/服务业增加值`
  - `服务业污水排放量/服务业增加值 废水排放/服务业增加值`
- 输出 Excel：固定两个 Sheet：
  - `熵权`：ID 列 + 各指标贡献列 + `综合指数`
  - `topsis`：ID 列 + `正理想解距离D+`、`负理想解距离D-`、`相对接近度C`、`排名`

### 1) `熵权topsis法 - 以省份为分组 -年份.py`

- 参数/默认值
  - `--input-file`：必填
  - `--output-file`：默认 `topsis_by_province_year.xlsx`
  - `--group-col`：默认 `省份`
  - `--year-col`：默认 `年份`
  - `--negative-indicators`：默认空（使用默认列表）
  - `--run-dir`：默认空（自动创建 run dir）
- 输出
  - 文件名：`topsis_by_province_year.xlsx`
  - Sheet：`熵权`、`topsis`
  - 关键列（ID 顺序）：`年份`、`省份`
  - 备注：按 `省份` 分组计算权重/得分；脚本**不**额外输出权重表/权重行。

### 2) `熵权topsis法---加入省份列.py`

- 参数/默认值
  - `--input-file`：必填
  - `--output-file`：默认 `topsis_with_region.xlsx`
  - `--year-col`：默认 `年份`
  - `--region-col`：默认 `省份`
  - `--negative-indicators`：默认空（使用默认列表）
  - `--run-dir`：默认空（自动创建 run dir）
- 输出
  - 文件名：`topsis_with_region.xlsx`
  - Sheet：`熵权`、`topsis`
  - 关键列（ID 顺序）：`年份`、`省份`（取 `--region-col`）
  - 备注：对全表计算一次（不分组）；脚本**不**追加权重行。

### 3) `熵权topsis法-单省份.py`

- 参数/默认值
  - `--input-file`：必填
  - `--output-file`：默认 `topsis_single.xlsx`
  - `--name-col`：默认 `学校名称`（若不存在则回退为第 1 列列名）
  - `--negative-indicators`：默认空（使用默认列表）
  - `--eps-shift`：默认 `0.01`
  - `--run-dir`：默认空（自动创建 run dir）
- 输出
  - 文件名：`topsis_single.xlsx`
  - Sheet：`熵权`、`topsis`
  - 关键列（ID 顺序）：`学校名称`（或回退后的第 1 列）
  - 备注：`熵权` Sheet 末尾会追加 1 行权重：第 1 个 ID 列为 `权重`，各指标列填入权重值。

### 4) `熵权topsis法-年份.py`

- 参数/默认值
  - `--input-file`：必填
  - `--output-file`：默认 `topsis_by_year.xlsx`
  - `--year-col`：默认 `年份`
  - `--negative-indicators`：默认空（使用默认列表）
  - `--run-dir`：默认空（自动创建 run dir）
- 输出
  - 文件名：`topsis_by_year.xlsx`
  - Sheet：`熵权`、`topsis`
  - 关键列（ID 顺序）：`年份`
  - 备注：`熵权` Sheet 末尾会追加 1 行权重：`年份` 列为 `权重`，各指标列填入权重值。

## 秩和检验（2 个脚本）

共同点：
- 统计阈值：`alpha=0.05`（脚本内固定，不暴露参数）

### 5) `秩和检验.py`（questions 模式）

- 参数/默认值
  - `--input-file`：必填
  - `--output-file`：默认 `rank_test_summary.xlsx`
  - `--question-cols`：必填（逗号分隔，如 `Q1,Q2,Q3`）
  - `--dv-col`：默认 `BIU`
  - `--run-dir`：默认空（自动创建 run dir）
- 输出
  - 文件名：`rank_test_summary.xlsx`
  - Sheet：`Summary`
  - 关键列：`Question`、`TestUsed`、`Statistic`、`pValue`、`Significant`、`Groups`

### 6) `秩和检验---分组.py`（group 模式）

- 参数/默认值
  - `--input-file`：必填
  - `--output-file`：默认 `group_rank_test_results.xlsx`
  - `--group-col`：默认 `区域`
  - `--run-dir`：默认空（自动创建 run dir）
- 输出
  - 文件名：`group_rank_test_results.xlsx`
  - Sheet：默认 `Sheet1`（未显式指定 `sheet_name`）
  - 关键列：`变量`、`检验方法`、`statistic`、`p值`
