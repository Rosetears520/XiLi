import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from common.cli_utils import enter_run_dir, prepare_output_path, require_input_path


def configure_visual_settings() -> None:
    """配置matplotlib和seaborn的可视化设置"""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.family': 'Microsoft YaHei',
        'font.sans-serif': ['Microsoft YaHei'],
        'axes.unicode_minus': False,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 12
    })


@dataclass
class ClusterAnalysisResult:
    """存储聚类分析结果的容器类"""
    optimal_k: int
    best_silhouette: float
    cluster_labels: np.ndarray
    cluster_centers_std: pd.DataFrame
    cluster_centers_orig: pd.DataFrame
    cluster_counts: pd.Series
    cluster_summary: pd.DataFrame
    pca_results: pd.DataFrame
    metrics: Dict[str, float]


class KMeansAnalyzer:
    """K-means聚类分析器"""

    def __init__(self, data_path: Path, output_dir: Path) -> None:
        """
        初始化分析器

        参数:
            data_path: 数据文件路径
            output_dir: 输出目录路径
        """
        self.data_path: Path = data_path
        self.output_dir: Path = output_dir
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[List[str]] = None
        self.scaler: StandardScaler = StandardScaler()
        self.data_scaled: Optional[np.ndarray] = None
        self.metrics_xy_data: Optional[pd.DataFrame] = None

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """加载并预处理数据"""
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            for encoding in ["utf-8", "gbk", "utf-8-sig"]:
                try:
                    self.data = pd.read_csv(self.data_path, encoding=encoding)
                    break
                except Exception:
                    continue
            if self.data is None:
                self.data = pd.read_csv(self.data_path)
        else:
            self.data = pd.read_excel(self.data_path)

        # 检查缺失值
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print("警告：数据包含缺失值：\n", missing[missing > 0])

        # 检查非数值型列
        non_numeric = self.data.select_dtypes(exclude=['number']).columns
        if len(non_numeric) > 0:
            print("警告：存在非数值型列：", list(non_numeric))

        print(f"数据加载成功，形状: {self.data.shape}")
        return self.data

    def _select_features(self) -> List[str]:
        """选择用于聚类的数值型特征"""
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            raise ValueError("数据中没有可用的数值型特征列。")
        print("使用特征列:", numeric_cols)
        return numeric_cols

    def _get_scaled_data(self) -> np.ndarray:
        """对选定特征进行标准化处理，并缓存结果"""
        if self.features is None:
            self.features = self._select_features()
        self.data_scaled = self.scaler.fit_transform(self.data[self.features])
        return self.data_scaled

    def determine_optimal_k(
            self, max_k: int = 10, use_advanced_metrics: bool = True
    ) -> Tuple[int, Dict[str, float]]:
        """
        确定最佳聚类数

        参数:
            max_k: 最大聚类数
            use_advanced_metrics: 是否使用高级评估指标

        返回:
            (最佳聚类数, 评估指标字典)
        """
        print("开始确定最佳聚类数...")
        self.features = self._select_features()
        data_scaled = self._get_scaled_data()

        k_range = range(2, max_k + 1)
        metrics = {
            'sse': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }

        for k in tqdm(k_range, desc="计算不同k值的指标"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', init='k-means++')
            labels = kmeans.fit_predict(data_scaled)
            metrics['sse'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(data_scaled, labels))
            if use_advanced_metrics:
                metrics['calinski_harabasz'].append(calinski_harabasz_score(data_scaled, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(data_scaled, labels))

        self.metrics_xy_data = pd.DataFrame({
            "k": list(k_range),
            "sse": metrics["sse"],
            "轮廓系数": metrics["silhouette"],
            "Calinski-Harabasz": metrics["calinski_harabasz"],
            "Davies-Bouldin": metrics["davies_bouldin"]
        })

        self._plot_cluster_metrics(k_range, metrics)

        optimal_index = np.argmax(metrics['silhouette'])
        optimal_k = list(k_range)[optimal_index]
        best_metrics = {
            'silhouette': metrics['silhouette'][optimal_index],
            'calinski_harabasz': metrics['calinski_harabasz'][optimal_index] if use_advanced_metrics else None,
            'davies_bouldin': metrics['davies_bouldin'][optimal_index] if use_advanced_metrics else None
        }

        print(f"确定最佳聚类数: {optimal_k}")
        print(f"最佳轮廓系数: {best_metrics['silhouette']:.4f}")
        return optimal_k, best_metrics

    def _plot_cluster_metrics(self, k_range: range, metrics: Dict[str, List[float]]) -> None:
        """绘制聚类评估指标图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('聚类评估指标', fontsize=16)

        # SSE (肘部法则)
        axes[0, 0].plot(list(k_range), metrics['sse'], 'bo-')
        axes[0, 0].set_title('肘部法则 (SSE)')
        axes[0, 0].set_xlabel('聚类数 k')
        axes[0, 0].set_ylabel('SSE')
        axes[0, 0].grid(True)

        # 轮廓系数
        axes[0, 1].plot(list(k_range), metrics['silhouette'], 'go-')
        axes[0, 1].set_title('轮廓系数')
        axes[0, 1].set_xlabel('聚类数 k')
        axes[0, 1].set_ylabel('轮廓系数')
        axes[0, 1].grid(True)

        # Calinski-Harabasz指数
        if metrics.get('calinski_harabasz'):
            axes[1, 0].plot(list(k_range), metrics['calinski_harabasz'], 'ro-')
            axes[1, 0].set_title('Calinski-Harabasz指数')
            axes[1, 0].set_xlabel('聚类数 k')
            axes[1, 0].set_ylabel('CH指数')
            axes[1, 0].grid(True)

        # Davies-Bouldin指数
        if metrics.get('davies_bouldin'):
            axes[1, 1].plot(list(k_range), metrics['davies_bouldin'], 'mo-')
            axes[1, 1].set_title('Davies-Bouldin指数')
            axes[1, 1].set_xlabel('聚类数 k')
            axes[1, 1].set_ylabel('DB指数')
            axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = self.output_dir / 'cluster_metrics.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"聚类评估指标图已保存: {plot_path}")

    def perform_clustering(self, optimal_k: int) -> ClusterAnalysisResult:
        """执行K-means聚类，并生成分析结果"""
        print(f"开始执行K-means聚类 (k={optimal_k})...")
        data_scaled = self._get_scaled_data()

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto', init='k-means++')
        self.kmeans = kmeans  # 保存模型
        cluster_labels = kmeans.fit_predict(data_scaled)

        metrics = {
            'silhouette': silhouette_score(data_scaled, cluster_labels),
            'calinski_harabasz': calinski_harabasz_score(data_scaled, cluster_labels),
            'davies_bouldin': davies_bouldin_score(data_scaled, cluster_labels)
        }

        result = self._analyze_clusters(data_scaled, cluster_labels, kmeans, metrics)
        print("聚类分析完成!")
        return result

    def _analyze_clusters(
            self,
            data_scaled: np.ndarray,
            cluster_labels: np.ndarray,
            kmeans: KMeans,
            metrics: Dict[str, float]
    ) -> ClusterAnalysisResult:
        """分析聚类结果并生成各类输出"""
        self.data['cluster'] = cluster_labels

        cluster_centers_std = pd.DataFrame(kmeans.cluster_centers_, columns=self.features)
        cluster_centers_orig = pd.DataFrame(
            self.scaler.inverse_transform(kmeans.cluster_centers_),
            columns=self.features
        )
        cluster_counts = self.data['cluster'].value_counts().sort_index()
        cluster_summary = self.data.groupby('cluster')[self.features].mean()
        pca_results = self._perform_pca_analysis(data_scaled, cluster_labels)

        self._save_results(
            cluster_centers_std,
            cluster_centers_orig,
            cluster_counts,
            cluster_summary,
            pca_results,
            metrics
        )

        return ClusterAnalysisResult(
            optimal_k=kmeans.n_clusters,
            best_silhouette=metrics['silhouette'],
            cluster_labels=cluster_labels,
            cluster_centers_std=cluster_centers_std,
            cluster_centers_orig=cluster_centers_orig,
            cluster_counts=cluster_counts,
            cluster_summary=cluster_summary,
            pca_results=pca_results,
            metrics=metrics
        )

    def _perform_pca_analysis(self, data_scaled: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
        """执行PCA降维，并生成聚类分布散点图"""
        print("执行PCA降维分析...")
        if data_scaled.ndim != 2 or data_scaled.shape[1] < 2:
            print("提示：特征数 < 2，跳过 PCA 降维与散点图输出。")
            return pd.DataFrame()
        if data_scaled.shape[0] < 2:
            print("提示：样本数 < 2，跳过 PCA 降维与散点图输出。")
            return pd.DataFrame()
        pca = PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(data_scaled)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        pca_df['cluster'] = cluster_labels

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            hue='cluster',
            palette='viridis',
            alpha=0.8,
            s=100,
            edgecolor='w'
        )
        explained_var = pca.explained_variance_ratio_
        plt.xlabel(f'PC1 ({explained_var[0] * 100:.1f}%)')
        plt.ylabel(f'PC2 ({explained_var[1] * 100:.1f}%)')
        plt.title('PCA降维后的聚类分布')
        plt.legend(title='聚类')

        plot_path = self.output_dir / 'pca_cluster_plot.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"PCA聚类图已保存: {plot_path}")
        return pca_df

    def _save_results(
            self,
            centers_std: pd.DataFrame,
            centers_orig: pd.DataFrame,
            counts: pd.Series,
            summary: pd.DataFrame,
            pca_results: pd.DataFrame,
            metrics: Dict[str, float]
    ) -> None:
        """将所有分析结果保存到文件"""
        # 保存Excel数据
        excel_path = self.output_dir / 'analysis_results.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            self.data.to_excel(writer, sheet_name='聚类后数据', index=False)
            centers_std.to_excel(writer, sheet_name='标准化聚类中心', index=False)
            centers_orig.to_excel(writer, sheet_name='原始聚类中心', index=False)
            counts.to_frame('样本数量').to_excel(writer, sheet_name='各簇样本数量')
            summary.to_excel(writer, sheet_name='各簇特征均值')
            pca_results.to_excel(writer, sheet_name='PCA结果', index=False)
            pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']) \
                .to_excel(writer, sheet_name='评估指标')
            if self.metrics_xy_data is not None:
                self.metrics_xy_data.to_excel(writer, sheet_name='聚类评估指标 数据', index=False)

        print(f"所有分析数据已保存到: {excel_path}")

        # 保存文本报告
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("K-means聚类分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. 聚类评估指标:\n")
            for name, value in metrics.items():
                f.write(f"{name.replace('_', ' ').title()}: {value:.4f}\n")

            f.write("\n2. 各聚类样本数量:\n")
            f.write(self.data['cluster'].value_counts().sort_index().to_string() + "\n")

            f.write("\n3. 聚类中心(原始尺度)特征:\n")
            f.write(pd.DataFrame(
                self.scaler.inverse_transform(self.kmeans.cluster_centers_),
                columns=self.features
            ).to_string() + "\n")

            f.write("\n4. 各聚类特征均值:\n")
            f.write(self.data.groupby('cluster')[self.features].mean().to_string() + "\n")

            f.write("\n5. PCA分析结果:\n")
            if pca_results is None or pca_results.empty or not {"PC1", "PC2"}.issubset(set(pca_results.columns)):
                f.write("PCA 已跳过：特征数或样本数不足。\n")
            else:
                f.write(f"主成分1解释方差: {pca_results['PC1'].var():.4f}\n")
                f.write(f"主成分2解释方差: {pca_results['PC2'].var():.4f}\n")

        print(f"分析报告已保存到: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K-means 聚类分析工具")
    parser.add_argument(
        "--input-file",
        default=None,
        help="输入 csv 和 excel 数据文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认: outputs/kmeans）",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="聚类数上限（默认: 10）",
    )
    parser.add_argument(
        "--use-advanced-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用高级评估指标（默认: True）",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="运行工作目录（默认: runs/<timestamp_uuid>/）",
    )
    return parser


def main(
    input_file: Path,
    output_dir: Path,
    max_k: int = 10,
    use_advanced_metrics: bool = True,
) -> None:
    """主函数"""
    configure_visual_settings()

    analyzer = KMeansAnalyzer(input_file, output_dir)
    analyzer.load_data()
    optimal_k, _ = analyzer.determine_optimal_k(
        max_k=max_k,
        use_advanced_metrics=use_advanced_metrics,
    )
    results = analyzer.perform_clustering(optimal_k)

    print("=" * 50)
    print("聚类分析成功完成!")
    print(f"最佳聚类数: {results.optimal_k}")
    print(f"轮廓系数: {results.best_silhouette:.4f}")
    print("各簇样本数量:\n", results.cluster_counts.to_string())


def cli(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        enter_run_dir(args.run_dir)
        input_file = require_input_path(
            args.input_file,
            "输入 csv 和 excel 数据文件",
            allowed_suffixes={".xlsx", ".xls", ".csv"},
        )
        output_dir = prepare_output_path(args.output_dir, "outputs/kmeans")
        if output_dir.suffix:
            raise ValueError(f"输出目录应为文件夹路径：{output_dir}")
        if output_dir.exists() and output_dir.is_file():
            raise ValueError(f"输出目录不可为文件：{output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        main(
            input_file=input_file,
            output_dir=output_dir,
            max_k=args.max_k,
            use_advanced_metrics=args.use_advanced_metrics,
        )
    except Exception as exc:
        print(f"分析过程中出错: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
