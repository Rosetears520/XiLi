# GUI acceptance checklist (Gradio)

This document defines the acceptance criteria for consolidating `gradio_toolbox.py` into only two GUI entry points
(TOPSIS / RankTest), while still covering all branch behaviors via `group_cols` and `year_col`.

Status: checklist/spec only; the implementation work is tracked in `TODO.md` (section C).

---

# GUI 验收清单（`gradio_toolbox.py` 两入口：TOPSIS / 秩和检验）

本文件用于定义 GUI 端到端验收的目标状态与测试用例矩阵，确保：
- `gradio_toolbox.py` 最终仅保留 2 个 GUI 入口（TOPSIS / 秩和检验）
- 通过输入不同 `group_cols` / `year_col`（以及 RankTest 的 `mode`）覆盖全部分支

相关规范（作为验收基准）：
- 分支判定 + 同名去重：`concepts_group_cols_year_col.md`
- 多列分组语义：`semantics_multi_group_cols.md`
- TOPSIS 输出规范：`output_spec_entropy_topsis.md`
- RankTest 统一接口：`ranktest_interface_spec.md`


## 1) 目标状态（Acceptance Goals）

### 1.1 GUI 入口收敛

- `gradio_toolbox.py` 仅保留两个入口（Tab 或等价 UI）：
  - `TOPSIS`（熵权 TOPSIS）
  - `RankTest`（秩和检验）
- 其余旧脚本仍保留 CLI/文件名用于兼容，但不再作为 GUI 入口暴露。

### 1.2 统一运行产物目录

- 每次运行落盘到 `runs/<timestamp_uuid>/`，并可下载全部产物（结果 Excel、日志、图片等）。


## 2) TOPSIS Tab：分支覆盖矩阵（`group_cols` / `year_col`）

分支选择规则以 `concepts_group_cols_year_col.md` 为准。验收用例至少覆盖以下组合：

| 用例 | `group_cols` | `year_col` | 预期分支 |
|---|---|---|---|
| T0 | 空 | 空 | 全表一次（不分组） |
| T1 | 空 | 非空 | 仅年份分支（`year_col` 仅作为 ID 保留） |
| T2 | 非空（1 或多列） | 空 | 仅分组分支（按 `group_cols` 联合分组） |
| T3 | 非空（1 或多列） | 非空 | 分组+年份分支（按 `group_cols` 联合分组；`year_col` 仅作为 ID 保留） |
| T4 | 包含 `year_col`（同名） | 非空 | 同名去重后走等价分支（可能退化为 T1） |

验收要点：
- 多列联合分组（2+）：组别键与排序、NA 处理遵循 `semantics_multi_group_cols.md`。
- 输出结构：熵权表/topsis 表的 ID 列顺序、去重与权重输出遵循 `output_spec_entropy_topsis.md`。
- `group_cols` 与 `year_col` 同名时自动去重，不出现重复列名与重复 ID 列。


## 3) RankTest Tab：模式覆盖矩阵（`mode` + `group_cols`）

以 `ranktest_interface_spec.md` 为准，至少覆盖：

| 用例 | `mode` | 关键输入 | 预期行为 |
|---|---|---|---|
| R1 | `questions` | `question_cols` + `dv_col` | 输出题目级汇总（列：`Question/TestUsed/Statistic/pValue/Significant/Groups`） |
| R2 | `group` | `group_cols`（单列） | 输出变量级结果（列：`变量/检验方法/statistic/p值`） |
| R3 | `group` | `group_cols`（多列） | 按多列联合分组构造组别，仍输出变量级结果 |

互斥/校验：
- `mode=questions` 不接受 `group_cols`
- `mode=group` 不接受 `question_cols`（以及 `dv_col`）


## 4) 手工验收步骤（建议）

1) 运行：`python gradio_toolbox.py`
2) TOPSIS Tab：用同一份最小样例数据分别跑 T0~T4；核对输出表结构与列顺序。
3) RankTest Tab：用同一份最小样例数据分别跑 R1~R3；核对输出列与组别数量边界行为（组数 <2、不做检验等）。
4) 核对每次运行产物均落在 `runs/<timestamp_uuid>/` 且可下载。

