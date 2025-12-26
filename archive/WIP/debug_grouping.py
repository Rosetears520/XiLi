#!/usr/bin/env python3
"""Debug script to understand TOPSIS grouping behavior in detail."""

import pandas as pd
import numpy as np
from xili.entropy_topsis import run_entropy_topsis

def create_test_data():
    """Create test data with clear numeric and non-numeric columns."""
    np.random.seed(42)
    n_rows = 8  # Smaller for easier debugging

    data = {
        'Year': [2020, 2021, 2020, 2021, 2020, 2021, 2020, 2021],
        'Province': ['Beijing', 'Beijing', 'Shanghai', 'Shanghai', 'Guangdong', 'Guangdong', 'Jiangsu', 'Jiangsu'],
        'City': ['Beijing', 'Beijing', 'Shanghai', 'Shanghai', 'Guangzhou', 'Guangzhou', 'Nanjing', 'Nanjing'],
        'Indicator1': np.random.rand(n_rows) * 100,
        'Indicator2': np.random.rand(n_rows) * 50,
        'Indicator3': np.random.rand(n_rows) * 200,
    }
    return pd.DataFrame(data)

def debug_group_only():
    """Debug group-only scenario step by step."""
    print("=== Debug: Group Only ===")
    df = create_test_data()
    print(f"Original df shape: {df.shape}")
    print(f"Original df columns: {list(df.columns)}")
    print(f"Original df:\n{df}")
    print()

    try:
        result = run_entropy_topsis(
            df,
            group_cols=['Province'],
            eps_shift=0.01
        )

        print(f"Metric cols: {result.metric_cols}")
        print(f"Weights shape: {result.weights.shape}")
        print(f"Weights columns: {list(result.weights.columns)}")
        print(f"Weights:\n{result.weights}")
        print(f"Entropy table shape: {result.entropy.shape}")
        print(f"Entropy table columns: {list(result.entropy.columns)}")
        print(f"First few rows of entropy:\n{result.entropy.head()}")
        print()
    except Exception as e:
        print(f"Error in group-only: {e}")
        import traceback
        traceback.print_exc()
        print()

def debug_group_with_year():
    """Debug group + year scenario step by step."""
    print("=== Debug: Group + Year ===")
    df = create_test_data()

    try:
        result = run_entropy_topsis(
            df,
            group_cols=['Province'],
            year_col='Year',
            eps_shift=0.01
        )

        print(f"Metric cols: {result.metric_cols}")
        print(f"Weights shape: {result.weights.shape}")
        print(f"Weights columns: {list(result.weights.columns)}")
        print(f"Weights:\n{result.weights}")
        print(f"Entropy table shape: {result.entropy.shape}")
        print(f"Entropy table columns: {list(result.entropy.columns)}")
        print(f"First few rows of entropy:\n{result.entropy.head()}")
        print()
    except Exception as e:
        print(f"Error in group+year: {e}")
        import traceback
        traceback.print_exc()
        print()

def debug_with_explicit_id_cols():
    """Debug with explicit ID columns to see behavior."""
    print("=== Debug: With Explicit ID Columns ===")
    df = create_test_data()

    try:
        result = run_entropy_topsis(
            df,
            id_cols=['City'],  # Explicitly specify City as ID
            group_cols=['Province'],
            eps_shift=0.01
        )

        print(f"Metric cols: {result.metric_cols}")
        print(f"Weights shape: {result.weights.shape}")
        print(f"Weights columns: {list(result.weights.columns)}")
        print(f"Weights:\n{result.weights}")
        print(f"Entropy table shape: {result.entropy.shape}")
        print(f"Entropy table columns: {list(result.entropy.columns)}")
        print(f"First few rows of entropy:\n{result.entropy.head()}")
        print()
    except Exception as e:
        print(f"Error with explicit ID cols: {e}")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    debug_group_only()
    debug_group_with_year()
    debug_with_explicit_id_cols()
