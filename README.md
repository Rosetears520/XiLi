# XiLiSuite（溪狸析理）

XiLiSuite 是一套专注于数据分析与梳理的工具套件，强调易使用、可复用、可复现。
像一只狸花猫在数据的小溪边捡光拾花，把凌乱一点点理顺，最后让答案在喧嚣里也清亮可见。

## 功能概览

- 统一入口：GUI + CLI（同一套参数/输出逻辑）
- 核心分析：熵权 TOPSIS、秩和检验（两组/多组）
- 预处理：线性插值、年度平均、年均增长率填补
- 图像与文本：黑白图片、文档分词、词云图
- NLP：LDA 困惑度与一致性评估、LDA 模型、SnowNLP 模型训练与情感分析

## 快速开始（源码）

本项目使用 `uv` 管理依赖（建议 CPython 3.9+）。

```bash
uv venv
uv sync
```

启动 GUI（默认端口 1221；端口占用时会尝试清理并仍使用 1221）：

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

## 打包（Windows / Nuitka）

需要安装 Visual Studio Build Tools（C++ x64 工具链 + Windows SDK）。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build/build_nuitka.ps1
```

打包产物在 `dist/main.dist/`，将整个目录压缩后分发即可（不要只发单个 exe）。

## 输出与复现

- 默认输出在 `runs/`（每次运行一个独立时间戳目录），便于复现与归档
- GUI 日志在终端输出；关闭终端窗口会退出程序

## 项目结构

- `main.py`: 统一入口（GUI + CLI 子命令）
- `scripts/gui/gradio_toolbox.py`: GUI（Gradio）
- `scripts/tools/`: 各工具的可复用实现（均提供 `cli(argv)`）
- `xili/`: 可复用算法与统一 CLI（熵权 TOPSIS、秩和检验）
- `common/`: 通用 CLI、路径、运行上下文与输出封装
- `docs/`: 使用说明与补充文档
- `tests/`: 自动化测试
