#!/usr/bin/env python3
"""Test script to verify TOPSIS grouping functionality."""

import pandas as pd
import numpy as np
from xili.entropy_topsis import run_entropy_topsis

def create_test_data():
    """Create test data with grouping and year columns."""
    np.random.seed(42)
    n_rows = 20

    data = {
        '年份': [2020, 2021] * 10,
        '省份': ['北京', '上海', '广东', '江苏'] * 5,
        '城市': ['北京', '上海', '广州', '南京'] * 5,
        '指标1': np.random.rand(n_rows) * 100,
        '指标2': np.random.rand(n_rows) * 50,
        '指标3': np.random.rand(n_rows) * 200,
    }
    return pd.DataFrame(data)

def test_group_only():
    """Test group-only scenario."""
    print("=== Test: Group Only ===")
    df = create_test_data()

    result = run_entropy_topsis(
        df,
        group_cols=['省份'],
        eps_shift=0.01
    )

    print(f"Metric cols: {result.metric_cols}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"Entropy table columns: {list(result.entropy.columns)}")
    print(f"First few rows of weights:\n{result.weights.head()}")
    print()

def test_group_year():
    """Test group + year scenario."""
    print("=== Test: Group + Year ===")
    df = create_test_data()

    result = run_entropy_topsis(
        df,
        group_cols=['省份'],
        year_col='年份',
        eps_shift=0.01
    )

    print(f"Metric cols: {result.metric_cols}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"Entropy table columns (first 10): {list(result.entropy.columns[:10])}")
    print(f"First few rows of weights:\n{result.weights.head()}")
    print()

def test_multi_group():
    """Test multi-group scenario."""
    print("=== Test: Multi-Group ===")
    df = create_test_data()

    result = run_entropy_topsis(
        df,
        group_cols=['省份', '城市'],
        year_col='年份',
        eps_shift=0.01
    )

    print(f"Metric cols: {result.metric_cols}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"Entropy table columns (first 10): {list(result.entropy.columns[:10])}")
    print(f"First few rows of weights:\n{result.weights.head()}")
    print()

def test_same_name_dedup():
    """Test same-name deduplication."""
    print("=== Test: Same Name Dedup ===")
    df = create_test_data()

    # Pass year_col also in group_cols - should be deduplicated
    result = run_entropy_topsis(
        df,
        group_cols=['省份', '年份'],  # 年份 also in group_cols
        year_col='年份',
        eps_shift=0.01
    )

    print(f"Metric cols: {result.metric_cols}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"First few rows of weights:\n{result.weights.head()}")
    print()

if __name__ == "__main__":
    test_group_only()
    test_group_year()
    test_multi_group()
    test_same_name_dedup()
