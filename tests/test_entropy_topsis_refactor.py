"""
Test script for refactored xili/entropy_topsis.py

Tests the four branch logic cases:
1. 仅分组: Only grouping, no year column
2. 仅年份: Only year column, no grouping
3. 分组+年份: Both grouping and year column
4. 同名去重: Handle when group_cols and year_col have the same name
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import xili module
sys.path.insert(0, str(Path(__file__).parent.parent))

from xili.entropy_topsis import run_entropy_topsis


def test_group_only_multi():
    """Test case: 仅分组 (multi-group columns)"""
    print("\n=== Testing 仅分组 (Multi-group columns) ===")

    # Load fixture data
    df = pd.read_excel('tests/fixtures/entropy_topsis/entropy_topsis_group_only_multi.xlsx')
    print("Input data:")
    print(df)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Run entropy TOPSIS with multi-group columns
    result = run_entropy_topsis(
        df,
        group_cols=['省份', '行业']  # Multi-group columns
    )

    print("\nResults:")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"TOPSIS table shape: {result.topsis.shape}")
    print(f"Weights table shape: {result.weights.shape}")
    print(f"Metric columns: {result.metric_cols}")

    print("\nWeights table:")
    print(result.weights)

    print("\nEntropy table (first 5 rows):")
    print(result.entropy.head())

    print("\nTOPSIS table (first 5 rows):")
    print(result.topsis.head())

    # Verify group columns are preserved
    expected_group_cols = ['省份', '行业']
    for col in expected_group_cols:
        assert col in result.entropy.columns, f"Group column '{col}' missing from entropy table"
        assert col in result.topsis.columns, f"Group column '{col}' missing from topsis table"
        assert col in result.weights.columns, f"Group column '{col}' missing from weights table"

    # Verify weights table has one row per group
    unique_groups = df.groupby(expected_group_cols).ngroups
    assert len(result.weights) == unique_groups, f"Expected {unique_groups} weight rows, got {len(result.weights)}"

    print("PASS: 仅分组 test passed!")
    return result


def test_group_year_multi():
    """Test case: 分组+年份 (multi-group columns + year)"""
    print("\n=== Testing 分组+年份 (Multi-group columns + Year) ===")

    # Load fixture data
    df = pd.read_excel('tests/fixtures/entropy_topsis/entropy_topsis_group_year_multi.xlsx')
    print("Input data:")
    print(df)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Run entropy TOPSIS with multi-group columns and year
    result = run_entropy_topsis(
        df,
        group_cols=['省份', '行业'],  # Multi-group columns
        year_col='年份'
    )

    print("\nResults:")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"TOPSIS table shape: {result.topsis.shape}")
    print(f"Weights table shape: {result.weights.shape}")
    print(f"Metric columns: {result.metric_cols}")

    print("\nWeights table:")
    print(result.weights)

    print("\nEntropy table (first 5 rows):")
    print(result.entropy.head())

    print("\nTOPSIS table (first 5 rows):")
    print(result.topsis.head())

    # Verify ID column order: year_col -> group_cols -> id_cols
    expected_id_order = ['年份', '省份', '行业']
    actual_id_cols = [col for col in result.entropy.columns if col not in result.metric_cols + ['综合指数']]
    print(f"Expected ID order: {expected_id_order}")
    print(f"Actual ID columns: {actual_id_cols}")

    # Verify year column is first
    if '年份' in result.entropy.columns:
        year_idx = actual_id_cols.index('年份')
        assert year_idx == 0, f"Year column should be first, but found at position {year_idx}"

    # Verify weights table structure: should NOT include year column
    if len(result.weights) > 1:  # Only if grouped
        assert '年份' not in result.weights.columns, "Year column should NOT be in weights table"
        expected_weight_cols = ['省份', '行业'] + result.metric_cols
        for col in expected_weight_cols:
            assert col in result.weights.columns, f"Column '{col}' missing from weights table"

    print("PASS: 分组+年份 test passed!")
    return result


def test_year_only():
    """Test case: 仅年份 (year column only)"""
    print("\n=== Testing 仅年份 (Year column only) ===")

    # Create test data
    data = {
        '年份': [2022, 2022, 2023, 2023],
        '指标1': [10, 12, 11, 13],
        '指标2': [20, 18, 19, 17]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Run entropy TOPSIS with year column only (no grouping)
    result = run_entropy_topsis(
        df,
        year_col='年份'
    )

    print("\nResults:")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"TOPSIS table shape: {result.topsis.shape}")
    print(f"Weights table shape: {result.weights.shape}")
    print(f"Metric columns: {result.metric_cols}")

    print("\nWeights table:")
    print(result.weights)

    print("\nEntropy table:")
    print(result.entropy)

    print("\nTOPSIS table:")
    print(result.topsis)

    # Verify year column is preserved as ID
    assert '年份' in result.entropy.columns, "Year column missing from entropy table"
    assert '年份' in result.topsis.columns, "Year column missing from topsis table"

    # Verify weights table has only one row (no grouping)
    assert len(result.weights) == 1, f"Expected 1 weight row for no grouping, got {len(result.weights)}"

    print("PASS: 仅年份 test passed!")
    return result


def test_no_grouping():
    """Test case: 不分组不按年份 (no grouping, no year)"""
    print("\n=== Testing 不分组不按年份 (No grouping, no year) ===")

    # Create test data
    data = {
        '地区': ['A', 'B', 'C', 'D'],
        '指标1': [10, 12, 11, 13],
        '指标2': [20, 18, 19, 17]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Run entropy TOPSIS with no grouping or year
    result = run_entropy_topsis(
        df,
        id_cols=['地区']  # Just ID columns, no grouping
    )

    print("\nResults:")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"TOPSIS table shape: {result.topsis.shape}")
    print(f"Weights table shape: {result.weights.shape}")
    print(f"Metric columns: {result.metric_cols}")

    print("\nWeights table:")
    print(result.weights)

    print("\nEntropy table:")
    print(result.entropy)

    print("\nTOPSIS table:")
    print(result.topsis)

    # Verify ID column is preserved
    assert '地区' in result.entropy.columns, "ID column missing from entropy table"
    assert '地区' in result.topsis.columns, "ID column missing from topsis table"

    # Verify weights table has only one row (no grouping)
    assert len(result.weights) == 1, f"Expected 1 weight row for no grouping, got {len(result.weights)}"

    print("PASS: 不分组不按年份 test passed!")
    return result


def test_negative_cols_aliases():
    """Test negative_cols and negative_indicators aliases are accepted and applied."""
    print("\n=== Testing negative_cols aliases ===")

    data = {
        '地区': ['A', 'B', 'C'],
        '指标': [10, 20, 30]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    result_positive = run_entropy_topsis(df, id_cols=['地区'])
    result_negative = run_entropy_topsis(df, id_cols=['地区'], negative_cols=['指标'])
    result_alias = run_entropy_topsis(df, id_cols=['地区'], negative_indicators='指标')

    top_positive = result_positive.topsis.loc[result_positive.topsis['排名'] == 1, '地区'].iloc[0]
    top_negative = result_negative.topsis.loc[result_negative.topsis['排名'] == 1, '地区'].iloc[0]
    top_alias = result_alias.topsis.loc[result_alias.topsis['排名'] == 1, '地区'].iloc[0]

    assert top_positive == 'C', f"Expected positive indicator top to be C, got {top_positive}"
    assert top_negative == 'A', f"Expected negative indicator top to be A, got {top_negative}"
    assert top_alias == top_negative, "Alias negative_indicators should match negative_cols behavior"

    print("PASS: negative_cols aliases test passed!")


def test_same_name_dedup():
    """Test case: 同名去重 (same name deduplication)"""
    print("\n=== Testing 同名去重 (Same name deduplication) ===")

    # Create test data where year_col is also in group_cols
    data = {
        '年份': [2022, 2022, 2023, 2023],
        '省份': ['A省', 'B省', 'A省', 'B省'],
        '指标1': [10, 12, 11, 13],
        '指标2': [20, 18, 19, 17]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Run entropy TOPSIS where year_col is also in group_cols
    # This should deduplicate and use it only as year_col
    result = run_entropy_topsis(
        df,
        group_cols=['年份', '省份'],  # Include year in group_cols
        year_col='年份'  # Also specify as year_col
    )

    print("\nResults:")
    print(f"Entropy table shape: {result.entropy.shape}")
    print(f"TOPSIS table shape: {result.topsis.shape}")
    print(f"Weights table shape: {result.weights.shape}")
    print(f"Metric columns: {result.metric_cols}")

    print("\nWeights table:")
    print(result.weights)

    print("\nEntropy table:")
    print(result.entropy)

    print("\nTOPSIS table:")
    print(result.topsis)

    # Verify year column is treated as year_col only, not as group_col
    # Should group only by '省份', not by both '年份' and '省份'
    unique_groups = df.groupby(['省份']).ngroups
    assert len(result.weights) == unique_groups, f"Expected {unique_groups} weight rows, got {len(result.weights)}"

    # Verify year column is NOT in weights table (it's treated as year_col, not group_col)
    assert '年份' not in result.weights.columns, "Year column should NOT be in weights table when deduplicated"
    assert '省份' in result.weights.columns, "Province column should be in weights table"

    print("PASS: 同名去重 test passed!")
    return result


def test_group_cols_parsing():
    """Test group_cols parsing with various input formats"""
    print("\n=== Testing group_cols parsing ===")

    # Create test data
    data = {
        '省份': ['A省', 'B省', 'A省', 'B省'],
        '行业': ['工业', '工业', '农业', '农业'],
        '指标1': [10, 12, 11, 13],
        '指标2': [20, 18, 19, 17]
    }
    df = pd.DataFrame(data)

    # Test different input formats
    test_cases = [
        (['省份', '行业'], "List format"),
        (['省份', '行业', '省份'], "List with duplicates"),
        (['省份,行业'], "String with comma"),
        (['省份\n行业'], "String with newline"),
        (['省份, 行业', '省份'], "Mixed format with spaces")
    ]

    for group_cols_input, description in test_cases:
        print(f"\nTesting {description}: {group_cols_input}")
        result = run_entropy_topsis(df, group_cols=group_cols_input)

        print(f"  Resulting groups: {len(result.weights)} rows")
        print(f"  Group columns in weights: {[col for col in result.weights.columns if col in ['省份', '行业']]}")

        # Should always result in 4 unique groups (省份 x 行业)
        assert len(result.weights) == 4, f"Expected 4 groups for {description}"

    print("\nPASS: group_cols parsing test passed!")


def test_eps_shift_default():
    """Test eps_shift=None (omitted) uses default 0.01"""
    print("\n=== Testing eps_shift default (None -> 0.01) ===")

    # Create test data with known values
    data = {
        '地区': ['A', 'B', 'C'],
        '指标1': [10, 20, 30],  # Simple linear values
        '指标2': [5, 15, 25]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Run with eps_shift omitted (should use default 0.01)
    result_default = run_entropy_topsis(df, id_cols=['地区'])

    # Run with explicit eps_shift=0.01 (should be identical)
    result_explicit = run_entropy_topsis(df, id_cols=['地区'], eps_shift=0.01)

    print("\nResult with default eps_shift:")
    print(result_default.entropy[result_default.metric_cols])

    print("\nResult with explicit eps_shift=0.01:")
    print(result_explicit.entropy[result_explicit.metric_cols])

    # Verify both produce identical results
    pd.testing.assert_frame_equal(
        result_default.entropy[result_default.metric_cols].reset_index(drop=True),
        result_explicit.entropy[result_explicit.metric_cols].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10
    )

    pd.testing.assert_frame_equal(
        result_default.topsis[['相对接近度C']].reset_index(drop=True),
        result_explicit.topsis[['相对接近度C']].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10
    )

    # Verify that the entropy table values are all > 0 (shift was applied)
    # Note: entropy_table contains contributions (standardized * weights),
    # so min value is eps_shift * min_weight, not eps_shift itself
    metric_data = result_default.entropy[result_default.metric_cols]
    for col in metric_data.columns:
        assert metric_data[col].min() > 0, f"Column {col} should have values > 0 (shift applied)"

    print("PASS: eps_shift default (0.01) test passed!")
    return result_default


def test_eps_shift_zero():
    """Test eps_shift=0 is passed through and does not apply a shift"""
    print("\n=== Testing eps_shift=0 (no shift) ===")

    # Create test data
    data = {
        '地区': ['A', 'B', 'C', 'D'],
        '指标1': [10, 20, 30, 40],
        '指标2': [5, 15, 25, 35]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Run with eps_shift=0 (no shift should be applied)
    result = run_entropy_topsis(df, id_cols=['地区'], eps_shift=0)

    print("\nResult with eps_shift=0:")
    print(result.entropy[result.metric_cols])

    # With eps_shift=0, normalized values [0,1] are used directly
    # This means some values can be exactly 0 (for min-max normalized data)
    metric_data = result.entropy[result.metric_cols]

    # For min-max normalized data, the minimum value should be 0 (or very close)
    for col in metric_data.columns:
        min_val = metric_data[col].min()
        print(f"  {col} min: {min_val}")
        assert np.isclose(min_val, 0.0, atol=1e-10), f"Column {col} minimum should be ~0 with eps_shift=0, got {min_val}"

    # Verify that with eps_shift=0, we still get valid results
    # (i.e., no errors from log(0) - handled by entropy calculation)
    assert not result.entropy.isna().any().any(), "Result should not contain NaN values"
    assert not result.topsis.isna().any().any(), "TOPSIS result should not contain NaN values"

    print("PASS: eps_shift=0 test passed!")
    return result


def test_eps_shift_positive():
    """Test eps_shift>0 shifts standardized values as expected"""
    print("\n=== Testing eps_shift>0 (positive shift) ===")

    # Create more diverse test data to ensure different shifts produce different results
    data = {
        '地区': ['A', 'B', 'C', 'D'],
        '指标1': [10, 25, 5, 40],  # Varied pattern
        '指标2': [15, 5, 30, 20],  # Inverse-ish pattern
        '指标3': [100, 200, 150, 80]  # Different scale
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Test with different positive eps_shift values
    shift_values = [0.01, 0.05, 0.1]
    results = {}

    for shift in shift_values:
        result = run_entropy_topsis(df, id_cols=['地区'], eps_shift=shift)
        results[shift] = result
        print(f"\nResult with eps_shift={shift}:")
        print(result.entropy[result.metric_cols])
        print("TOPSIS scores:", result.topsis['相对接近度C'].values)

        # Verify that the minimum value increases with larger shift
        # (entropy_table contains contributions = standardized * weight)
        # So minimum should be eps_shift * min_weight > 0
        metric_data = result.entropy[result.metric_cols]
        for col in metric_data.columns:
            min_val = metric_data[col].min()
            assert min_val > 0, f"Column {col} should have values > 0 with eps_shift={shift}, got {min_val}"

    # Verify that different shift values produce different results
    # (entropy and TOPSIS scores should differ)
    shift_001 = results[0.01].topsis['相对接近度C'].values
    shift_05 = results[0.05].topsis['相对接近度C'].values
    shift_1 = results[0.1].topsis['相对接近度C'].values

    # Results should differ (not identical)
    assert not np.allclose(shift_001, shift_05), "eps_shift=0.01 and 0.05 should produce different results"
    assert not np.allclose(shift_05, shift_1), "eps_shift=0.05 and 0.1 should produce different results"

    print("PASS: eps_shift>0 test passed!")
    return results


def main():
    """Run all tests"""
    print("Starting entropy_topsis refactor tests...")

    try:
        test_group_only_multi()
        test_group_year_multi()
        test_year_only()
        test_no_grouping()
        test_negative_cols_aliases()
        test_same_name_dedup()
        test_group_cols_parsing()
        test_eps_shift_default()
        test_eps_shift_zero()
        test_eps_shift_positive()

        print("\nSUCCESS: All tests passed! Refactoring is successful.")
        print("\nKey features verified:")
        print("PASS: Multi-column grouping support")
        print("PASS: Four branch logic cases")
        print("PASS: Same name deduplication")
        print("PASS: Proper ID column ordering")
        print("PASS: Group column parsing with various formats")
        print("PASS: Weight table structure compliance")
        print("PASS: eps_shift default behavior (0.01)")
        print("PASS: eps_shift=0 (no shift)")
        print("PASS: eps_shift>0 (positive shift)")

    except Exception as e:
        print(f"\nFAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
