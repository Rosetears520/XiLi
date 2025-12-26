#!/usr/bin/env python3
"""Comprehensive test to validate TOPSIS grouping implementation works correctly."""

import pandas as pd
import numpy as np
from xili.entropy_topsis import run_entropy_topsis

def create_test_data_like_fixtures():
    """Create test data that mirrors the structure of test fixtures but with English names."""
    np.random.seed(42)

    # Create data similar to group+year fixture
    data = {
        'Year': [2022, 2022, 2022, 2023, 2023, 2023, 2023],
        'Province': ['A', 'A', 'B', 'A', 'B', 'B', 'A'],
        'Industry': ['Tech', 'Finance', 'Tech', 'Tech', 'Finance', 'Finance', 'Tech'],
        'Metric1': [10, 11, 9, 12, 8, 15, 13],
        'Metric2': [20, 19, 22, 18, 23, 17, 21],
    }
    return pd.DataFrame(data)

def test_group_only():
    """Test group-only scenario - should exclude grouping cols from metrics."""
    print("=== Test: Group Only ===")
    df = create_test_data_like_fixtures()

    result = run_entropy_topsis(
        df,
        group_cols=['Province', 'Industry'],
        eps_shift=0.01
    )

    print(f"Input columns: {list(df.columns)}")
    print(f"Group cols: ['Province', 'Industry']")
    print(f"Metric cols: {result.metric_cols}")
    print(f"Expected: ['Year', 'Metric1', 'Metric2'] (Year treated as metric since not specified as year_col)")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Sample weights:
{result.weights.head()}")

    # Verify the output structure follows specification
    expected_weights_cols = ['Province', 'Industry', 'Year', 'Metric1', 'Metric2']
    assert list(result.weights.columns) == expected_weights_cols, f"Expected {expected_weights_cols}, got {list(result.weights.columns)}"
    assert result.metric_cols == ['Year', 'Metric1', 'Metric2'], f"Expected ['Year', 'Metric1', 'Metric2'], got {result.metric_cols}"
    print("PASS: Group Only test passed")
    print()

def test_group_with_year():
    """Test group+year scenario - should exclude both grouping and year cols from metrics."""
    print("=== Test: Group + Year ===")
    df = create_test_data_like_fixtures()

    result = run_entropy_topsis(
        df,
        group_cols=['Province', 'Industry'],
        year_col='Year',
        eps_shift=0.01
    )

    print(f"Input columns: {list(df.columns)}")
    print(f"Group cols: ['Province', 'Industry']")
    print(f"Year col: 'Year'")
    print(f"Metric cols: {result.metric_cols}")
    print(f"Expected: ['Metric1', 'Metric2'] (Year excluded as year_col)")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Sample weights:
{result.weights.head()}")

    # Verify the output structure follows specification
    expected_weights_cols = ['Province', 'Industry', 'Metric1', 'Metric2']
    assert list(result.weights.columns) == expected_weights_cols, f"Expected {expected_weights_cols}, got {list(result.weights.columns)}"
    assert result.metric_cols == ['Metric1', 'Metric2'], f"Expected ['Metric1', 'Metric2'], got {result.metric_cols}"
    print("PASS: Group + Year test passed")
    print()

def test_same_name_dedup():
    """Test same-name deduplication."""
    print("=== Test: Same Name Dedup ===")
    df = create_test_data_like_fixtures()

    result = run_entropy_topsis(
        df,
        group_cols=['Province', 'Industry', 'Year'],  # Year also in group_cols
        year_col='Year',
        eps_shift=0.01
    )

    print(f"Input columns: {list(df.columns)}")
    print(f"Group cols: ['Province', 'Industry', 'Year'] (Year duplicated)")
    print(f"Year col: 'Year'")
    print(f"After dedup, effective group cols: ['Province', 'Industry']")
    print(f"Metric cols: {result.metric_cols}")
    print(f"Expected: ['Metric1', 'Metric2']")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Sample weights:
{result.weights.head()}")

    # Year should be deduplicated from group_cols and treated only as year_col
    expected_weights_cols = ['Province', 'Industry', 'Metric1', 'Metric2']
    assert list(result.weights.columns) == expected_weights_cols, f"Expected {expected_weights_cols}, got {list(result.weights.columns)}"
    assert result.metric_cols == ['Metric1', 'Metric2'], f"Expected ['Metric1', 'Metric2'], got {result.metric_cols}"
    print("PASS: Same Name Dedup test passed")
    print()

def test_with_explicit_id_cols():
    """Test with explicit ID columns."""
    print("=== Test: With Explicit ID Columns ===")
    df = create_test_data_like_fixtures()

    result = run_entropy_topsis(
        df,
        id_cols=['Industry'],  # Explicit ID column
        group_cols=['Province'],
        year_col='Year',
        eps_shift=0.01
    )

    print(f"Input columns: {list(df.columns)}")
    print(f"ID cols: ['Industry']")
    print(f"Group cols: ['Province']")
    print(f"Year col: 'Year'")
    print(f"Metric cols: {result.metric_cols}")
    print(f"Expected: ['Metric1', 'Metric2']")
    print(f"Weights columns: {list(result.weights.columns)}")
    print(f"Weights shape: {result.weights.shape}")
    print(f"Sample weights:
{result.weights.head()}")

    # Industry should be excluded from metrics as explicit ID column
    expected_weights_cols = ['Province', 'Metric1', 'Metric2']
    assert list(result.weights.columns) == expected_weights_cols, f"Expected {expected_weights_cols}, got {list(result.weights.columns)}"
    assert result.metric_cols == ['Metric1', 'Metric2'], f"Expected ['Metric1', 'Metric2'], got {result.metric_cols}"
    print("PASS: With Explicit ID Columns test passed")
    print()

def test_output_structure():
    """Test that output structure follows specification."""
    print("=== Test: Output Structure Compliance ===")
    df = create_test_data_like_fixtures()

    result = run_entropy_topsis(
        df,
        group_cols=['Province', 'Industry'],
        year_col='Year',
        id_cols=[],
        eps_shift=0.01
    )

    print("Checking entropy table structure...")
    # Entropy table should have ID columns in correct order: year_col -> group_cols -> metric_cols + 综合指数
    expected_entropy_prefix = ['Year', 'Province', 'Industry']
    entropy_prefix = list(result.entropy.columns[:3])
    assert entropy_prefix == expected_entropy_prefix, f"Expected entropy prefix {expected_entropy_prefix}, got {entropy_prefix}"
    assert '综合指数' in result.entropy.columns, "综合指数 should be in entropy table"
    print("PASS: Entropy table structure correct")

    print("Checking topsis table structure...")
    # TOPSIS table should have same ID columns followed by TOPSIS columns
    expected_topsis_prefix = ['Year', 'Province', 'Industry']
    topsis_prefix = list(result.topsis.columns[:3])
    expected_topsis_suffix = ['正理想解距离D+', '负理想解距离D-', '相对接近度C', '排名']
    topsis_suffix = list(result.topsis.columns[-4:])
    assert topsis_prefix == expected_topsis_prefix, f"Expected topsis prefix {expected_topsis_prefix}, got {topsis_prefix}"
    assert topsis_suffix == expected_topsis_suffix, f"Expected topsis suffix {expected_topsis_suffix}, got {topsis_suffix}"
    print("PASS: TOPSIS table structure correct")

    print("Checking weights table structure...")
    # Weights table should have group_cols followed by metric_cols
    expected_weights_prefix = ['Province', 'Industry']
    expected_weights_suffix = ['Metric1', 'Metric2']
    weights_prefix = list(result.weights.columns[:2])
    weights_suffix = list(result.weights.columns[-2:])
    assert weights_prefix == expected_weights_prefix, f"Expected weights prefix {expected_weights_prefix}, got {weights_prefix}"
    assert weights_suffix == expected_weights_suffix, f"Expected weights suffix {expected_weights_suffix}, got {weights_suffix}"
    # Year should NOT be in weights table (per specification: year_col is not part of weights primary key)
    assert 'Year' not in result.weights.columns, "Year should not be in weights table"
    print("PASS: Weights table structure correct")

    print("PASS: All output structure tests passed")
    print()

if __name__ == "__main__":
    print("Running comprehensive TOPSIS grouping tests...
")

    test_group_only()
    test_group_with_year()
    test_same_name_dedup()
    test_with_explicit_id_cols()
    test_output_structure()

    print("SUCCESS: All tests passed! TOPSIS grouping implementation is working correctly.")
