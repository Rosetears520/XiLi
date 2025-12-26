from __future__ import annotations

import argparse
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _iter_artifact_files(run_dir: Path) -> Iterable[Path]:
    for path in run_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(run_dir)
        parts = rel.parts
        if any(part.startswith(".") for part in parts):
            continue
        if parts and parts[0] == "inputs":
            continue
        yield path


def collect_artifacts(run_dir: str | Path) -> tuple[list[tuple[str, str]], list[str]]:
    run_path = Path(run_dir)
    images_with_captions: list[tuple[str, str]] = []
    download_files: list[str] = []
    seen: set[Path] = set()

    for path in _iter_artifact_files(run_path):
        if path in seen:
            continue
        seen.add(path)
        download_files.append(str(path))
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            images_with_captions.append((str(path), path.name))

    if len(download_files) > 1:
        zip_path = run_path / "all_outputs.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in seen:
                if path == zip_path:
                    continue
                rel = path.relative_to(run_path)
                zf.write(path, rel.as_posix())
        download_files.append(str(zip_path))

    return images_with_captions, download_files


def _resolve_file_path(file_obj: Optional[object]) -> Optional[Path]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    path = getattr(file_obj, "name", None)
    if not path:
        return None
    return Path(path)


def _safe_read_table(file_path: Path, max_rows: int = 200) -> Optional[pd.DataFrame]:
    try:
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path).head(max_rows)
        return pd.read_excel(file_path).head(max_rows)
    except Exception:
        return None


def _load_columns(file_obj: Optional[object]) -> list[str]:
    file_path = _resolve_file_path(file_obj)
    if not file_path:
        return []
    df = _safe_read_table(file_path, max_rows=1)
    if df is None:
        return []
    return [str(c).strip() for c in df.columns]


def _update_input_preview(file_obj: Optional[object]):
    import gradio as gr

    file_path = _resolve_file_path(file_obj)
    if not file_path:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False)
    if file_path.suffix.lower() in _IMAGE_SUFFIXES:
        return gr.update(value=None, visible=False), gr.update(value=str(file_path), visible=True)
    df = _safe_read_table(file_path)
    if df is None:
        return gr.update(value=None, visible=False), gr.update(value=None, visible=False)
    return gr.update(value=df, visible=True), gr.update(value=None, visible=False)


def _build_run_dir(tool_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _PROJECT_ROOT / "runs" / f"{stamp}_{tool_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_output_path(run_dir: Path, output_text: str, default_name: str) -> Path:
    text = (output_text or "").strip()
    if text:
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate
        if candidate.parent == Path("."):
            return (run_dir / candidate.name).resolve()
        return (run_dir / candidate).resolve()
    return (run_dir / default_name).resolve()


def _resolve_output_dir(run_dir: Path, output_text: str, default_name: str) -> Path:
    text = (output_text or "").strip()
    if text:
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate
        return (run_dir / candidate).resolve()
    return (run_dir / default_name).resolve()


def _run_entry(command: str, args: list[str]) -> int:
    from scripts.entry.main import main as entry_main

    print(f"[GUI] main.py {command} {' '.join(args)}", flush=True)
    original_cwd = os.getcwd()
    try:
        return entry_main([command] + args)
    finally:
        os.chdir(original_cwd)


def _finish_run(run_dir: Path, return_code: int, tool_name: str):
    import gradio as gr

    status = "完成" if return_code == 0 else "失败"
    images, files = collect_artifacts(run_dir)
    gallery = gr.update(value=images, visible=bool(images))
    downloads = gr.update(value=files, visible=bool(files))
    output_preview = _preview_outputs(files)
    output_update = gr.update(value=output_preview, visible=output_preview is not None)
    return f"{tool_name} {status}，输出目录：{run_dir}", gallery, downloads, output_update


def _preview_outputs(download_files: list[str]) -> Optional[pd.DataFrame]:
    for path_text in download_files:
        path = Path(path_text)
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            df = _safe_read_table(path)
            if df is not None:
                return df
    return None


def _build_demo():
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
    css = """
    body, .gradio-container {
        font-family: "Segoe UI", "Inter", "Arial", "Microsoft YaHei", sans-serif;
    }
    .title {font-size: 26px; font-weight: 700;}
    .section-title {font-size: 18px; font-weight: 600; margin-top: 6px;}
    """

    with gr.Blocks(title="XiLi 析理溪狸") as demo:
        gr.Markdown(
            "<div class='title'>XiLi 析理溪狸</div>"
        )
        with gr.Tabs():
            with gr.Tab("核心分析"):
                with gr.Accordion("熵权 TOPSIS", open=True):
                    with gr.Row():
                        with gr.Column(scale=3):
                            topsis_input = gr.File(label="输入 Excel/CSV")
                            topsis_output_file = gr.Textbox(label="输出文件名", value="entropy_topsis.xlsx")
                            topsis_group_cols = gr.Dropdown(label="分组列名（可多选）", choices=[], multiselect=True)
                            topsis_year_col = gr.Dropdown(label="年份列名（可选）", choices=[])
                            topsis_id_cols = gr.Dropdown(label="ID 列名（可多选）", choices=[], multiselect=True)
                            topsis_negative_cols = gr.Dropdown(label="负向指标列名（可多选）", choices=[], multiselect=True)
                            topsis_eps_shift = gr.Number(label="非负平移常数", value=0.01, precision=4)
                            topsis_append_weights = gr.Checkbox(label="追加权重行", value=False)
                            topsis_run = gr.Button("运行熵权 TOPSIS", variant="primary")
                            topsis_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            topsis_preview_table = gr.Dataframe(label="输入预览")
                            topsis_preview_image = gr.Image(label="图片预览", visible=False)
                            topsis_output_preview = gr.Dataframe(label="输出预览")
                            topsis_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            topsis_files = gr.Files(label="下载输出")

                with gr.Accordion("秩和检验", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            rank_input = gr.File(label="输入 Excel/CSV")
                            rank_output_file = gr.Textbox(label="输出文件名", value="rank_tests.xlsx")
                            rank_mode = gr.Radio(label="模式", choices=["questions", "group"], value="questions")
                            rank_question_cols = gr.Dropdown(label="问题列（可多选）", choices=[], multiselect=True)
                            rank_dv_col = gr.Dropdown(label="因变量列（可选）", choices=[])
                            rank_group_cols = gr.Dropdown(label="分组列（可多选）", choices=[], multiselect=True, visible=False)
                            rank_alpha = gr.Number(label="显著性水平", value=0.05, precision=4)
                            rank_run = gr.Button("运行秩和检验", variant="primary")
                            rank_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            rank_preview_table = gr.Dataframe(label="输入预览")
                            rank_preview_image = gr.Image(label="图片预览", visible=False)
                            rank_output_preview = gr.Dataframe(label="输出预览")
                            rank_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            rank_files = gr.Files(label="下载输出")

            with gr.Tab("可视化/图像"):
                with gr.Accordion("词云图", open=True):
                    with gr.Row():
                        with gr.Column(scale=3):
                            wc_input = gr.File(label="词频文本文件")
                            wc_output = gr.Textbox(label="输出图片名", value="wordcloud.png")
                            wc_font_path = gr.Textbox(
                                label="字体路径（填入路径）",
                                placeholder="例如 C:\\Windows\\Fonts\\msyh.ttc",
                            )
                            with gr.Accordion("高级选项", open=False):
                                wc_mask = gr.File(label="遮罩图片（可选）")
                                wc_background = gr.Textbox(label="背景颜色", value="white")
                                wc_width = gr.Number(label="宽度", value=1600, precision=0)
                                wc_height = gr.Number(label="高度", value=900, precision=0)
                                wc_scale = gr.Number(label="画布缩放比例", value=1.0, precision=2)
                                wc_dpi = gr.Number(label="输出 DPI", value=300, precision=0)
                                wc_prefer_horizontal = gr.Slider(
                                    label="水平排版概率",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.05,
                                )
                                wc_min_font_size = gr.Number(label="最小字体大小", value=4, precision=0)
                                wc_font_step = gr.Number(label="字体步长", value=1, precision=0)
                                wc_max_words = gr.Number(label="最大词数", value=500, precision=0)
                                wc_max_font_size = gr.Textbox(label="最大字体大小（可选）", value="")
                                wc_relative_scaling = gr.Number(label="词频-字体关联性", value=0.5, precision=2)
                                wc_color_func = gr.Dropdown(
                                    label="颜色函数",
                                    choices=["none", "gradient", "fixed"],
                                    value="none",
                                )
                                wc_colormap = gr.Textbox(label="Colormap", value="viridis")
                            wc_run = gr.Button("生成词云图", variant="primary")
                            wc_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            wc_preview_table = gr.Dataframe(label="输入预览")
                            wc_preview_image = gr.Image(label="图片预览", visible=False)
                            wc_output_preview = gr.Dataframe(label="输出预览")
                            wc_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            wc_files = gr.Files(label="下载输出")

                with gr.Accordion("黑白图片", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            bi_input = gr.File(label="输入图片")
                            bi_output = gr.Textbox(label="输出图片名", value="binary.png")
                            bi_threshold = gr.Slider(label="阈值", minimum=0, maximum=255, value=128, step=1)
                            bi_run = gr.Button("生成黑白图片", variant="primary")
                            bi_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            bi_preview_table = gr.Dataframe(label="输入预览")
                            bi_preview_image = gr.Image(label="图片预览", visible=False)
                            bi_output_preview = gr.Dataframe(label="输出预览")
                            bi_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            bi_files = gr.Files(label="下载输出")

                with gr.Accordion("文档分词", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            seg_input = gr.File(label="输入文件 (txt/docx/xlsx/csv)")
                            seg_output = gr.Textbox(label="分词结果输出名", value="segmentation.txt")
                            seg_wordfreq = gr.Textbox(label="词频输出名", value="segmentation_wordfreq.txt")
                            seg_text_col = gr.Textbox(label="文本列名", value="content")
                            seg_stopwords = gr.File(label="停用词表（可选）")
                            seg_userdict = gr.File(label="自定义词典（可选）")
                            seg_mode = gr.Radio(
                                label="分词模式",
                                choices=["precise", "full", "search"],
                                value="search",
                            )
                            seg_run = gr.Button("运行文档分词", variant="primary")
                            seg_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            seg_preview_table = gr.Dataframe(label="输入预览")
                            seg_preview_image = gr.Image(label="图片预览", visible=False)
                            seg_output_preview = gr.Dataframe(label="输出预览")
                            seg_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            seg_files = gr.Files(label="下载输出")

            with gr.Tab("描述统计/填补"):
                with gr.Accordion("平均增长率", open=True):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr_input = gr.File(label="输入 Excel/CSV")
                            gr_output = gr.Textbox(label="输出文件名", value="average_growth_rate_result.csv")
                            with gr.Accordion("高级选项", open=False):
                                gr_group_cols = gr.Dropdown(label="分组列（可多选）", choices=[], multiselect=True)
                                gr_sort_col = gr.Dropdown(label="排序列（可选）", choices=[])
                                gr_id_cols = gr.Dropdown(label="不填充列（可多选）", choices=[], multiselect=True)
                                gr_cols = gr.Dropdown(label="填充列（可多选；不选则自动）", choices=[], multiselect=True)
                                gr_round = gr.Checkbox(label="四舍五入结果", value=False)
                            gr_run = gr.Button("计算平均增长率", variant="primary")
                            gr_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            gr_preview_table = gr.Dataframe(label="输入预览")
                            gr_preview_image = gr.Image(label="图片预览", visible=False)
                            gr_output_preview = gr.Dataframe(label="输出预览")
                            gr_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            gr_files = gr.Files(label="下载输出")

                with gr.Accordion("线性插值", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            li_input = gr.File(label="输入 CSV/Excel")
                            li_output = gr.Textbox(label="输出文件名", value="linear_interpolation_result.csv")
                            with gr.Accordion("高级选项", open=False):
                                li_group_cols = gr.Dropdown(label="分组列（可多选）", choices=[], multiselect=True)
                                li_sort_col = gr.Dropdown(label="排序列（可选）", choices=[])
                                li_id_cols = gr.Dropdown(label="不插值列（可多选）", choices=[], multiselect=True)
                                li_cols = gr.Dropdown(label="插值列（可多选；不选则自动）", choices=[], multiselect=True)
                                li_limit_direction = gr.Dropdown(
                                    label="插值方向",
                                    choices=["forward", "both"],
                                    value="forward",
                                )
                                li_limit_area = gr.Dropdown(
                                    label="插值范围",
                                    choices=["inside", "none"],
                                    value="inside",
                                )
                                li_round = gr.Checkbox(label="四舍五入结果", value=False)
                            li_run = gr.Button("运行线性插值", variant="primary")
                            li_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            li_preview_table = gr.Dataframe(label="输入预览")
                            li_preview_image = gr.Image(label="图片预览", visible=False)
                            li_output_preview = gr.Dataframe(label="输出预览")
                            li_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            li_files = gr.Files(label="下载输出")

                with gr.Accordion("年份平均值", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            ya_input = gr.File(label="输入 Excel/CSV")
                            ya_output = gr.Textbox(label="输出文件名", value="yearly_averages.xlsx")
                            ya_cols = gr.Dropdown(label="计算列（可多选）", choices=[], multiselect=True)
                            ya_rename = gr.Textbox(label="列重命名（可选，格式：A=A_avg;B=B_avg）")
                            ya_run = gr.Button("计算年份平均值", variant="primary")
                            ya_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            ya_preview_table = gr.Dataframe(label="输入预览")
                            ya_preview_image = gr.Image(label="图片预览", visible=False)
                            ya_output_preview = gr.Dataframe(label="输出预览")
                            ya_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            ya_files = gr.Files(label="下载输出")

            with gr.Tab("聚类分析"):
                with gr.Accordion("K-means 聚类", open=True):
                    with gr.Row():
                        with gr.Column(scale=3):
                            km_input = gr.File(label="输入 Excel/CSV")
                            km_output_dir = gr.Textbox(label="输出目录名", value="kmeans")
                            km_max_k = gr.Slider(label="最大 K 值", minimum=2, maximum=20, value=10, step=1)
                            km_run = gr.Button("运行 K-means", variant="primary")
                            km_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            km_preview_table = gr.Dataframe(label="输入预览")
                            km_preview_image = gr.Image(label="图片预览", visible=False)
                            km_output_preview = gr.Dataframe(label="输出预览")
                            km_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            km_files = gr.Files(label="下载输出")

            with gr.Tab("主题模型/NLP"):
                with gr.Accordion("LDA困惑度和一致性", open=True):
                    with gr.Row():
                        with gr.Column(scale=3):
                            ldae_input = gr.File(label="输入分词文件")
                            ldae_output = gr.Textbox(label="输出文件名", value="lda_evaluation.xlsx")
                            ldae_min = gr.Slider(label="最小主题数", minimum=2, maximum=30, value=2, step=1)
                            ldae_max = gr.Slider(label="最大主题数", minimum=2, maximum=40, value=20, step=1)
                            ldae_passes = gr.Slider(label="训练次数(passes)", minimum=1, maximum=50, value=5, step=1)
                            ldae_run = gr.Button("计算困惑度和一致性", variant="primary")
                            ldae_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            ldae_preview_table = gr.Dataframe(label="输入预览")
                            ldae_preview_image = gr.Image(label="图片预览", visible=False)
                            ldae_output_preview = gr.Dataframe(label="输出预览")
                            ldae_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            ldae_files = gr.Files(label="下载输出")

                with gr.Accordion("LDA模型", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            lda_input = gr.File(label="输入分词文件")
                            lda_output_dir = gr.Textbox(label="输出目录名", value="lda_model")
                            lda_topics = gr.Slider(label="主题数", minimum=2, maximum=30, value=10, step=1)
                            with gr.Accordion("高级选项", open=False):
                                lda_no_below = gr.Number(label="no_below（去除低频词）", value=3, precision=0)
                                lda_no_above = gr.Number(label="no_above（去除高频词比例）", value=0.5, precision=2)
                                lda_passes = gr.Number(label="passes（训练轮数）", value=20, precision=0)
                                lda_alpha = gr.Textbox(label="alpha（文档-主题先验）", value="auto")
                                lda_eta = gr.Textbox(label="eta（主题-词先验）", value="auto")
                                lda_coherence = gr.Dropdown(
                                    label="coherence_measure（一致性评分标准）",
                                    choices=["c_v", "u_mass", "c_uci", "c_npmi"],
                                    value="c_v",
                                )
                                lda_min_prob = gr.Number(label="minimum_probability（最小概率阈值）", value=0.01, precision=4)
                            lda_run = gr.Button("训练 LDA 模型", variant="primary")
                            lda_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            lda_preview_table = gr.Dataframe(label="输入预览")
                            lda_preview_image = gr.Image(label="图片预览", visible=False)
                            lda_output_preview = gr.Dataframe(label="输出预览")
                            lda_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            lda_files = gr.Files(label="下载输出")

                with gr.Accordion("Snowlp训练模型", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            sn_train_neg = gr.File(label="负向文本文件")
                            sn_train_pos = gr.File(label="正向文本文件")
                            sn_train_output = gr.Textbox(label="输出模型名", value="sentiment.marshal")
                            sn_train_run = gr.Button("训练 SnowNLP 模型", variant="primary")
                            sn_train_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            sn_train_preview_table = gr.Dataframe(label="输入预览")
                            sn_train_preview_image = gr.Image(label="图片预览", visible=False)
                            sn_train_output_preview = gr.Dataframe(label="输出预览")
                            sn_train_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            sn_train_files = gr.Files(label="下载输出")

                with gr.Accordion("Snowlp情感分析", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            sn_model_file = gr.File(label="模型文件（推荐上传：sentiment.marshal.3）")
                            sn_model_path = gr.Textbox(
                                label="模型路径（可选）",
                                placeholder="例如 data/sentiment.marshal（Python3 下实际会读取 data/sentiment.marshal.3；优先使用上面的模型文件）",
                            )
                            sn_input = gr.File(label="输入 Excel/CSV")
                            sn_text_col = gr.Dropdown(label="文本列名", choices=[], value=None)
                            with gr.Accordion("高级选项", open=False):
                                sn_font = gr.Textbox(label="字体", value="Microsoft YaHei")
                                sn_bins = gr.Number(label="直方图 bins", value=50, precision=0)
                            sn_output = gr.Textbox(label="输出文件名", value="comments_with_sentiment.xlsx")
                            sn_run = gr.Button("运行情感分析", variant="primary")
                            sn_status = gr.Textbox(label="执行状态", max_lines=2)
                        with gr.Column(scale=4):
                            sn_preview_table = gr.Dataframe(label="输入预览")
                            sn_preview_image = gr.Image(label="图片预览", visible=False)
                            sn_output_preview = gr.Dataframe(label="输出预览")
                            sn_gallery = gr.Gallery(label="输出图片预览", columns=3, height=200)
                            sn_files = gr.Files(label="下载输出")

        def _toggle_rank_mode(mode: str):
            return gr.update(visible=mode == "questions"), gr.update(visible=mode == "questions"), gr.update(
                visible=mode == "group"
            )

        rank_mode.change(
            _toggle_rank_mode,
            inputs=rank_mode,
            outputs=[rank_question_cols, rank_dv_col, rank_group_cols],
        )

        def _bind_preview(file_input, table_output, image_output):
            file_input.change(_update_input_preview, inputs=file_input, outputs=[table_output, image_output])

        _bind_preview(topsis_input, topsis_preview_table, topsis_preview_image)
        _bind_preview(rank_input, rank_preview_table, rank_preview_image)
        _bind_preview(wc_input, wc_preview_table, wc_preview_image)
        _bind_preview(bi_input, bi_preview_table, bi_preview_image)
        _bind_preview(gr_input, gr_preview_table, gr_preview_image)
        _bind_preview(li_input, li_preview_table, li_preview_image)
        _bind_preview(ya_input, ya_preview_table, ya_preview_image)
        _bind_preview(km_input, km_preview_table, km_preview_image)
        _bind_preview(lda_input, lda_preview_table, lda_preview_image)
        _bind_preview(ldae_input, ldae_preview_table, ldae_preview_image)
        _bind_preview(sn_input, sn_preview_table, sn_preview_image)
        _bind_preview(seg_input, seg_preview_table, seg_preview_image)
        _bind_preview(sn_train_neg, sn_train_preview_table, sn_train_preview_image)
        _bind_preview(sn_train_pos, sn_train_preview_table, sn_train_preview_image)

        def _update_columns(file_obj):
            cols = _load_columns(file_obj)
            return (
                gr.update(choices=cols, value=[]),
                gr.update(choices=cols, value=None),
                gr.update(choices=cols, value=[]),
                gr.update(choices=cols, value=[]),
            )

        topsis_input.change(
            _update_columns,
            inputs=topsis_input,
            outputs=[topsis_group_cols, topsis_year_col, topsis_id_cols, topsis_negative_cols],
        )

        def _update_rank_columns(file_obj):
            cols = _load_columns(file_obj)
            return (
                gr.update(choices=cols, value=[]),
                gr.update(choices=cols, value=None),
                gr.update(choices=cols, value=[]),
            )

        rank_input.change(
            _update_rank_columns,
            inputs=rank_input,
            outputs=[rank_question_cols, rank_dv_col, rank_group_cols],
        )

        sn_input.change(lambda f: gr.update(choices=_load_columns(f), value=None), inputs=sn_input, outputs=sn_text_col)

        def _update_fill_columns(file_obj):
            cols = _load_columns(file_obj)
            return (
                gr.update(choices=cols, value=[]),
                gr.update(choices=cols, value=None),
                gr.update(choices=cols, value=[]),
                gr.update(choices=cols, value=[]),
            )

        gr_input.change(
            _update_fill_columns,
            inputs=gr_input,
            outputs=[gr_group_cols, gr_sort_col, gr_id_cols, gr_cols],
        )

        li_input.change(
            _update_fill_columns,
            inputs=li_input,
            outputs=[li_group_cols, li_sort_col, li_id_cols, li_cols],
        )

        ya_input.change(lambda f: gr.update(choices=_load_columns(f), value=[]), inputs=ya_input, outputs=ya_cols)

        def _run_topsis(
            file_obj,
            output_text,
            group_cols,
            year_col,
            id_cols,
            negative_cols,
            eps_shift,
            append_weights,
        ):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("topsis")
            output_path = _resolve_output_path(run_dir, output_text, "entropy_topsis.xlsx")
            args = [
                "--input-file",
                str(file_path),
                "--output-file",
                str(output_path),
                "--eps-shift",
                str(eps_shift),
                "--run-dir",
                str(run_dir),
            ]
            if group_cols:
                args += ["--group-cols", ",".join(group_cols)]
            if year_col:
                args += ["--year-col", year_col]
            if id_cols:
                args += ["--id-cols", ",".join(id_cols)]
            if negative_cols:
                args += ["--negative-indicators", ",".join(negative_cols)]
            if append_weights:
                args.append("--append-weights")
            code = _run_entry("topsis", args)
            return _finish_run(run_dir, code, "熵权 TOPSIS")

        topsis_run.click(
            _run_topsis,
            inputs=[
                topsis_input,
                topsis_output_file,
                topsis_group_cols,
                topsis_year_col,
                topsis_id_cols,
                topsis_negative_cols,
                topsis_eps_shift,
                topsis_append_weights,
            ],
            outputs=[topsis_status, topsis_gallery, topsis_files, topsis_output_preview],
        )

        def _run_ranktest(
            file_obj,
            output_text,
            mode,
            question_cols,
            dv_col,
            group_cols,
            alpha,
        ):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("ranktest")
            output_path = _resolve_output_path(run_dir, output_text, "rank_tests.xlsx")
            args = [
                "--input-file",
                str(file_path),
                "--output-file",
                str(output_path),
                "--mode",
                mode,
                "--alpha",
                str(alpha),
                "--run-dir",
                str(run_dir),
            ]
            if mode == "questions":
                if not dv_col:
                    return "questions 模式必须选择因变量列。", None, None, None
                if question_cols:
                    args += ["--question-cols", ",".join(question_cols)]
                if dv_col:
                    args += ["--dv-col", dv_col]
            else:
                if group_cols:
                    args += ["--group-cols", ",".join(group_cols)]
            code = _run_entry("ranktest", args)
            return _finish_run(run_dir, code, "秩和检验")

        rank_run.click(
            _run_ranktest,
            inputs=[
                rank_input,
                rank_output_file,
                rank_mode,
                rank_question_cols,
                rank_dv_col,
                rank_group_cols,
                rank_alpha,
            ],
            outputs=[rank_status, rank_gallery, rank_files, rank_output_preview],
        )

        def _run_wordcloud(
            file_obj,
            output_text,
            font_path,
            mask_obj,
            background_color,
            width,
            height,
            scale,
            dpi,
            prefer_horizontal,
            min_font_size,
            font_step,
            max_words,
            max_font_size_text,
            relative_scaling,
            color_func,
            colormap,
        ):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("wordcloud")
            output_path = _resolve_output_path(run_dir, output_text, "wordcloud.png")
            mask_path = _resolve_file_path(mask_obj)
            max_font_size = None
            if max_font_size_text:
                try:
                    max_font_size = int(max_font_size_text)
                except ValueError:
                    return "最大字体大小必须为整数或留空。", None, None, None
            args = ["--input-file", str(file_path), "--output-image", str(output_path)]
            if font_path:
                args += ["--font-path", font_path]
            if mask_path:
                args += ["--mask-image-path", str(mask_path)]
            if background_color:
                args += ["--background-color", str(background_color)]
            if width:
                args += ["--width", str(int(width))]
            if height:
                args += ["--height", str(int(height))]
            if scale:
                args += ["--scale", str(float(scale))]
            if dpi:
                args += ["--dpi", str(int(dpi))]
            if prefer_horizontal is not None:
                args += ["--prefer-horizontal", str(float(prefer_horizontal))]
            if min_font_size:
                args += ["--min-font-size", str(int(min_font_size))]
            if font_step:
                args += ["--font-step", str(int(font_step))]
            if max_words:
                args += ["--max-words", str(int(max_words))]
            if max_font_size is not None:
                args += ["--max-font-size", str(int(max_font_size))]
            if relative_scaling is not None:
                args += ["--relative-scaling", str(float(relative_scaling))]
            if color_func and str(color_func).lower() != "none":
                args += ["--color-func", str(color_func)]
            if colormap:
                args += ["--colormap", str(colormap)]
            args += ["--run-dir", str(run_dir)]
            code = _run_entry("wordcloud", args)
            return _finish_run(run_dir, code, "词云图")

        wc_run.click(
            _run_wordcloud,
            inputs=[
                wc_input,
                wc_output,
                wc_font_path,
                wc_mask,
                wc_background,
                wc_width,
                wc_height,
                wc_scale,
                wc_dpi,
                wc_prefer_horizontal,
                wc_min_font_size,
                wc_font_step,
                wc_max_words,
                wc_max_font_size,
                wc_relative_scaling,
                wc_color_func,
                wc_colormap,
            ],
            outputs=[wc_status, wc_gallery, wc_files, wc_output_preview],
        )

        def _run_binary(file_obj, output_text, threshold):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("binary_image")
            output_path = _resolve_output_path(run_dir, output_text, "binary.png")
            args = [
                "--input-file",
                str(file_path),
                "--output-file",
                str(output_path),
                "--threshold",
                str(int(threshold)),
            ]
            args += ["--run-dir", str(run_dir)]
            code = _run_entry("binary_image", args)
            return _finish_run(run_dir, code, "黑白图片")

        bi_run.click(
            _run_binary,
            inputs=[bi_input, bi_output, bi_threshold],
            outputs=[bi_status, bi_gallery, bi_files, bi_output_preview],
        )

        def _run_growth_configurable(
            file_obj,
            output_text,
            group_cols,
            sort_col,
            id_cols,
            cols,
            round_values,
        ):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            if not group_cols:
                return "请先选择分组列（至少 1 个）。", None, None, None
            run_dir = _build_run_dir("growth_rate")
            output_path = _resolve_output_path(run_dir, output_text, "average_growth_rate_result.csv")
            args = ["--input-file", str(file_path), "--output-file", str(output_path), "--run-dir", str(run_dir)]
            if group_cols:
                args += ["--group-cols", ",".join(group_cols)]
            if sort_col:
                args += ["--sort-col", sort_col]
            if id_cols:
                args += ["--id-cols", ",".join(id_cols)]
            if cols:
                args += ["--cols", ",".join(cols)]
            if round_values:
                args.append("--round")
            code = _run_entry("growth_rate", args)
            return _finish_run(run_dir, code, "平均增长率")

        gr_run.click(
            _run_growth_configurable,
            inputs=[gr_input, gr_output, gr_group_cols, gr_sort_col, gr_id_cols, gr_cols, gr_round],
            outputs=[gr_status, gr_gallery, gr_files, gr_output_preview],
        )

        def _run_linear_interpolation(
            file_obj,
            output_text,
            group_cols,
            sort_col,
            id_cols,
            cols,
            limit_direction,
            limit_area,
            round_values,
        ):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            if not group_cols:
                return "请先选择分组列（至少 1 个）。", None, None, None
            run_dir = _build_run_dir("linear_interpolation")
            output_path = _resolve_output_path(run_dir, output_text, "linear_interpolation_result.csv")
            args = [
                "--input-file",
                str(file_path),
                "--output-file",
                str(output_path),
                "--run-dir",
                str(run_dir),
            ]
            if group_cols:
                args += ["--group-cols", ",".join(group_cols)]
            if sort_col:
                args += ["--sort-col", sort_col]
            if id_cols:
                args += ["--id-cols", ",".join(id_cols)]
            if cols:
                args += ["--cols", ",".join(cols)]
            if limit_direction:
                args += ["--limit-direction", str(limit_direction)]
            if limit_area:
                args += ["--limit-area", str(limit_area)]
            if round_values:
                args.append("--round")
            code = _run_entry("linear_interpolation", args)
            return _finish_run(run_dir, code, "线性插值")

        li_run.click(
            _run_linear_interpolation,
            inputs=[
                li_input,
                li_output,
                li_group_cols,
                li_sort_col,
                li_id_cols,
                li_cols,
                li_limit_direction,
                li_limit_area,
                li_round,
            ],
            outputs=[li_status, li_gallery, li_files, li_output_preview],
        )

        def _run_yearly_average(file_obj, output_text, cols, rename_text):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("yearly_average")
            output_path = _resolve_output_path(run_dir, output_text, "yearly_averages.xlsx")
            args = ["--input-file", str(file_path), "--output-file", str(output_path), "--run-dir", str(run_dir)]
            if cols:
                args += ["--cols", ",".join(cols)]
            if rename_text:
                for pair in rename_text.split(";"):
                    pair = pair.strip()
                    if not pair:
                        continue
                    args += ["--rename", pair]
            code = _run_entry("yearly_average", args)
            return _finish_run(run_dir, code, "年份平均值")

        ya_run.click(
            _run_yearly_average,
            inputs=[ya_input, ya_output, ya_cols, ya_rename],
            outputs=[ya_status, ya_gallery, ya_files, ya_output_preview],
        )

        def _run_kmeans(file_obj, output_text, max_k):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("kmeans")
            output_dir = _resolve_output_dir(run_dir, output_text, "kmeans")
            args = [
                "--input-file",
                str(file_path),
                "--output-dir",
                str(output_dir),
                "--max-k",
                str(int(max_k)),
                "--run-dir",
                str(run_dir),
            ]
            code = _run_entry("kmeans", args)
            return _finish_run(run_dir, code, "K-means 聚类")

        km_run.click(
            _run_kmeans,
            inputs=[km_input, km_output_dir, km_max_k],
            outputs=[km_status, km_gallery, km_files, km_output_preview],
        )

        def _run_lda_model(
            file_obj,
            output_text,
            topics,
            no_below,
            no_above,
            passes,
            alpha,
            eta,
            coherence_measure,
            minimum_probability,
        ):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("lda_model")
            output_dir = _resolve_output_dir(run_dir, output_text, "lda_model")
            args = [
                "--input-file",
                str(file_path),
                "--output-dir",
                str(output_dir),
                "--num-topics",
                str(int(topics)),
            ]
            if no_below is not None:
                args += ["--no-below", str(int(no_below))]
            if no_above is not None:
                args += ["--no-above", str(float(no_above))]
            if passes is not None:
                args += ["--passes", str(int(passes))]
            if alpha:
                args += ["--alpha", str(alpha)]
            if eta:
                args += ["--eta", str(eta)]
            if coherence_measure:
                args += ["--coherence-measure", str(coherence_measure)]
            if minimum_probability is not None:
                args += ["--minimum-probability", str(float(minimum_probability))]
            args += ["--run-dir", str(run_dir)]
            code = _run_entry("lda_model", args)
            return _finish_run(run_dir, code, "LDA 模型")

        lda_run.click(
            _run_lda_model,
            inputs=[
                lda_input,
                lda_output_dir,
                lda_topics,
                lda_no_below,
                lda_no_above,
                lda_passes,
                lda_alpha,
                lda_eta,
                lda_coherence,
                lda_min_prob,
            ],
            outputs=[lda_status, lda_gallery, lda_files, lda_output_preview],
        )

        def _run_lda_eval(file_obj, output_text, min_topics, max_topics, passes):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("lda_eval")
            output_path = _resolve_output_path(run_dir, output_text, "lda_evaluation.xlsx")
            output_chart = _resolve_output_path(run_dir, "lda_evaluation.png", "lda_evaluation.png")
            args = [
                "--input-file",
                str(file_path),
                "--output-excel",
                str(output_path),
                "--output-chart",
                str(output_chart),
                "--topic-min",
                str(int(min_topics)),
                "--topic-max",
                str(int(max_topics)),
                "--passes",
                str(int(passes)),
                "--headless",
            ]
            args += ["--run-dir", str(run_dir)]
            code = _run_entry("lda_eval", args)
            return _finish_run(run_dir, code, "LDA困惑度和一致性")

        ldae_run.click(
            _run_lda_eval,
            inputs=[ldae_input, ldae_output, ldae_min, ldae_max, ldae_passes],
            outputs=[ldae_status, ldae_gallery, ldae_files, ldae_output_preview],
        )

        def _run_snownlp_train(neg_file, pos_file, output_text):
            neg_path = _resolve_file_path(neg_file)
            pos_path = _resolve_file_path(pos_file)
            if not neg_path or not pos_path:
                return "请先选择正/负向文本文件。", None, None, None
            run_dir = _build_run_dir("snowlp_train")
            output_path = _resolve_output_path(run_dir, output_text, "sentiment.marshal")
            args = [
                "--negative-file",
                str(neg_path),
                "--positive-file",
                str(pos_path),
                "--output-model",
                str(output_path),
            ]
            args += ["--run-dir", str(run_dir)]
            code = _run_entry("snowlp_train", args)
            return _finish_run(run_dir, code, "SnowNLP 训练")

        sn_train_run.click(
            _run_snownlp_train,
            inputs=[sn_train_neg, sn_train_pos, sn_train_output],
            outputs=[sn_train_status, sn_train_gallery, sn_train_files, sn_train_output_preview],
        )

        def _run_snownlp_analysis(model_file_obj, model_path_text, file_obj, text_col, font, bins, output_text):
            file_path = _resolve_file_path(file_obj)
            model_file_path = _resolve_file_path(model_file_obj)
            model_path_value = (model_path_text or "").strip()
            model_path = str(model_file_path) if model_file_path else (model_path_value or "")
            if not file_path or not model_path:
                return "请先提供模型文件/模型路径，并选择输入文件。", None, None, None
            run_dir = _build_run_dir("snowlp_analysis")
            output_path = _resolve_output_path(run_dir, output_text, "comments_with_sentiment.xlsx")
            args = [
                "--model-path",
                model_path,
                "--input-file",
                str(file_path),
                "--output-excel",
                str(output_path),
                "--headless",
            ]
            if text_col:
                args += ["--text-col", str(text_col)]
            if font:
                args += ["--chinese-font", str(font)]
            if bins is not None:
                args += ["--histogram-bins", str(int(bins))]
            args += ["--run-dir", str(run_dir)]
            code = _run_entry("snowlp_analysis", args)
            return _finish_run(run_dir, code, "SnowNLP 情感分析")

        sn_run.click(
            _run_snownlp_analysis,
            inputs=[sn_model_file, sn_model_path, sn_input, sn_text_col, sn_font, sn_bins, sn_output],
            outputs=[sn_status, sn_gallery, sn_files, sn_output_preview],
        )

        def _run_segmentation(
            file_obj,
            output_text,
            wordfreq_text,
            text_column,
            stopwords_obj,
            userdict_obj,
            mode,
        ):
            file_path = _resolve_file_path(file_obj)
            if not file_path:
                return "请先选择输入文件。", None, None, None
            run_dir = _build_run_dir("word_segmentation")
            output_path = _resolve_output_path(run_dir, output_text, "segmentation.txt")
            wordfreq_path = _resolve_output_path(run_dir, wordfreq_text, "segmentation_wordfreq.txt")
            stopwords_path = _resolve_file_path(stopwords_obj)
            userdict_path = _resolve_file_path(userdict_obj)
            args = [
                "--input-file",
                str(file_path),
                "--output-file",
                str(output_path),
                "--wordfreq-output-file",
                str(wordfreq_path),
                "--text-column",
                str(text_column or "content"),
                "--mode",
                str(mode),
                "--run-dir",
                str(run_dir),
            ]
            if stopwords_path:
                args += ["--stopwords-file", str(stopwords_path)]
            if userdict_path:
                args += ["--userdict-file", str(userdict_path)]
            code = _run_entry("word_segmentation", args)
            return _finish_run(run_dir, code, "文档分词")

        seg_run.click(
            _run_segmentation,
            inputs=[
                seg_input,
                seg_output,
                seg_wordfreq,
                seg_text_col,
                seg_stopwords,
                seg_userdict,
                seg_mode,
            ],
            outputs=[seg_status, seg_gallery, seg_files, seg_output_preview],
        )

    return demo, theme, css


def main(argv: list[str] | None = None) -> int:
    try:
        import gradio as gr  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - keep CLI robust
        print(f"GUI 启动失败: {exc}", file=sys.stderr)
        return 1

    def _can_bind(host: str, port: int) -> bool:
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, int(port)))
            return True
        except OSError:
            return False

    def _kill_processes_on_port(port: int) -> list[int]:
        import subprocess

        try:
            output = subprocess.check_output(["netstat", "-ano", "-p", "tcp"], text=True, errors="ignore")
        except Exception:
            output = subprocess.check_output(["netstat", "-ano"], text=True, errors="ignore")

        pids: set[int] = set()
        needle = f":{int(port)}"
        for line in output.splitlines():
            if needle not in line:
                continue
            if "LISTENING" not in line.upper():
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                pid = int(parts[-1])
            except Exception:
                continue
            pids.add(pid)

        killed: list[int] = []
        for pid in sorted(pids):
            try:
                subprocess.check_call(
                    ["taskkill", "/PID", str(pid), "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                killed.append(pid)
            except Exception:
                continue
        return killed

    def _ensure_port_1221(host: str, port: int, *, force: bool) -> None:
        import time

        if _can_bind(host, port):
            return
        if not force:
            raise OSError(f"端口 {port} 被占用（未启用自动清理）。")

        print(f"[GUI] 端口 {port} 被占用，尝试清理占用进程…", file=sys.stderr, flush=True)
        killed = _kill_processes_on_port(port)
        if killed:
            print(f"[GUI] 已尝试结束进程 PID: {', '.join(map(str, killed))}", file=sys.stderr, flush=True)

        for _ in range(30):
            if _can_bind(host, port):
                return
            time.sleep(0.2)

        raise OSError(f"端口 {port} 清理失败，仍被占用。")

    parser = argparse.ArgumentParser(prog="gradio_toolbox")
    parser.add_argument("--server-port", type=int, default=1221, help="Gradio server port")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server host")
    parser.add_argument("--share", action="store_true", help="Enable sharing")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument(
        "--force-port",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="端口被占用时，尝试清理占用进程并继续使用指定端口（默认: True）",
    )
    args = parser.parse_args(argv)

    demo, theme, css = _build_demo()
    try:
        _ensure_port_1221(str(args.server_name), int(args.server_port), force=bool(args.force_port))
        demo.launch(
            server_name=args.server_name,
            server_port=int(args.server_port),
            share=args.share,
            inbrowser=not args.no_browser,
            theme=theme,
            css=css,
        )
    except Exception as exc:
        print(f"GUI 启动失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
