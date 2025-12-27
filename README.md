# XiLiSuite（溪狸析理）

[![License](https://img.shields.io/github/license/Rosetears520/XiLi)](https://github.com/Rosetears520/XiLi)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

XiLiSuite 是一套专注于**数据分析与梳理**的工具套件，强调易使用、可复用、可复现。  
像一只狸花猫在数据的小溪边捡光拾花，把凌乱一点点理顺，让答案在喧嚣里也清亮可见。  

![XiLiSuite Logo](assets/logo.png)

## 功能概览

- **数据处理与统计分析**：线性插值、年度平均、年均增长率、K-Means 聚类、分组处理与检验等
- **图像与文本工具**：图片二值化（黑白化）、文档分词、词云生成
- **NLP / 主题模型**：LDA 训练与评估（困惑度/一致性）、SnowNLP 训练与情感分析

## 快速开始

### Windows

推荐前往 [Releases](https://github.com/Rosetears520/XiLi/releases) 页面下载 exe 使用。

### Mac / Linux

本项目使用 `uv` 管理依赖（建议 Python 3.12）。

```bash
uv venv
uv sync
```

启动 GUI（默认端口 `1221`；端口占用时会尝试清理并仍使用 `1221`）：

```bash
uv run -- python main.py gui
```

CLI 示例：

```bash
uv run -- python main.py topsis --help
uv run -- python main.py ranktest --help
uv run -- python main.py wordcloud --help
```

运行测试：

```bash
uv run -- pytest -q
```

## 功能脚本清单（当前仓库）

> 说明：以下为仓库根目录可见的功能脚本。长期建议以 `main.py` 作为统一入口（CLI/GUI），其余脚本更多用于独立运行、调试、以及逐步迁移整合。

### 数据处理与统计分析

* `kmeans.py` / `K-means.py`：K-Means 聚类（可能存在重复版本，建议后续收敛为一个）
* `yearly_average.py`：年度平均值计算
* `average_growth_rate.py`：年均增长率计算/填补
* `linear_interpolation.py`：线性插值（缺失值处理）
* `comprehensive_grouping_test.py`：分组综合测试脚本（偏验证/实验）
* `test_grouping.py`：分组逻辑测试脚本（偏验证/实验）

### 图像与文本工具

* `binary_image.py`：图片二值化/黑白化
* `word_segmentation.py`：文档分词
* `word_cloud.py`：词云图生成

### NLP / 主题模型

* `lda_model.py`：LDA 主题模型训练/推断
* `lda_evaluation.py`：LDA 评估（困惑度/一致性等）
* `snownlp_train.py`：SnowNLP 训练
* `snownlp_sentiment.py`：SnowNLP 情感分析

## 打包（Windows / Nuitka）

需要安装 Visual Studio Build Tools（C++ x64 工具链 + Windows SDK）。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build/build_nuitka.ps1
```

打包产物在 `dist/main.dist/`，将整个目录压缩后分发即可（不要只发单个 exe）。

## 输出与复现

* 默认输出在 `runs/`（每次运行一个独立时间戳目录），便于复现与归档
* GUI 日志在终端输出；关闭终端窗口会退出程序

## 项目结构

> 以仓库根目录为准（可能随重构调整）。

* `main.py`：统一入口（GUI + CLI 子命令）
* `gradio_toolbox.py`：GUI（Gradio）
* `xili/`：可复用算法与统一 CLI（如 熵权 TOPSIS、秩和检验）
* `common/`：通用 CLI、路径、运行上下文与输出封装
* `tests/`：自动化测试
* `assets/`：资源文件（图标/示例等）
* `archive/` / `legacy/`：历史内容与迁移保留（可逐步整理瘦身）

## 常见问题

* **GUI 起不来 / 端口被占用**：默认使用 `1221`，请先关闭占用该端口的进程，或检查是否有残留运行实例。
* **Windows 打包失败**：确认已安装 VS Build Tools（C++ x64 工具链 + Windows SDK），再运行 Nuitka 构建脚本。
* **关闭终端 GUI 就退出**：这是预期行为（当前 GUI 依赖终端生命周期）。

---

## 关于项目

* **项目主页**: [https://github.com/Rosetears520/XiLi](https://github.com/Rosetears520/XiLi)
* **作者博客**: [https://rosetears.cn](https://rosetears.cn) 
* **开源协议**: MIT License
* **问题反馈**: 请通过 [Issues](https://github.com/Rosetears520/XiLi/issues) 提交反馈
