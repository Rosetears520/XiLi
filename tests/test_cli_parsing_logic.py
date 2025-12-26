import pytest
import re
from typing import Optional, List, Tuple

def normalize_group_year_cols(
    year_col: Optional[str],
    group_cols_str: str,
    group_col_fallback: Optional[str] = None
) -> Tuple[Optional[str], List[str]]:
    # 1. Normalize year_col
    y = (year_col or "").strip()
    y = y if y else None

    # 2. Normalize group_cols
    # Split by comma or newline
    raw_groups = re.split(r'[,\n]', group_cols_str)
    
    # Strip and filter empty
    groups = [g.strip() for g in raw_groups if g.strip()]
    
    # Handle compatibility with single group-col fallback
    if not groups and group_col_fallback:
        groups = [group_col_fallback.strip()]
        groups = [g for g in groups if g]

    # Deduplicate while preserving order
    seen = set()
    deduped_groups = []
    for g in groups:
        if g not in seen:
            deduped_groups.append(g)
            seen.add(g)
    
    # 3. Remove year_col from group_cols if present
    if y and y in seen:
        deduped_groups = [g for g in deduped_groups if g != y]
        
    return y, deduped_groups

def test_normalize_basic():
    y, g = normalize_group_year_cols("Year", "Province,Industry")
    assert y == "Year"
    assert g == ["Province", "Industry"]

def test_normalize_newline():
    y, g = normalize_group_year_cols("Year", "Province\nIndustry")
    assert y == "Year"
    assert g == ["Province", "Industry"]

def test_normalize_mixed_separator():
    y, g = normalize_group_year_cols("Year", "Province, Industry\n City")
    assert y == "Year"
    assert g == ["Province", "Industry", "City"]

def test_normalize_dedup():
    y, g = normalize_group_year_cols("Year", "Province, Province, Industry")
    assert y == "Year"
    assert g == ["Province", "Industry"]

def test_normalize_year_in_groups():
    y, g = normalize_group_year_cols("Year", "Province, Year, Industry")
    assert y == "Year"
    assert g == ["Province", "Industry"]

def test_normalize_fallback():
    y, g = normalize_group_year_cols(None, "", "Province")
    assert y is None
    assert g == ["Province"]

def test_normalize_fallback_ignored_if_groups_provided():
    y, g = normalize_group_year_cols(None, "Province,Industry", "Other")
    assert y is None
    assert g == ["Province", "Industry"]

def test_normalize_empty():
    y, g = normalize_group_year_cols("", "  ", None)
    assert y is None
    assert g == []

def test_normalize_only_year():
    y, g = normalize_group_year_cols("Year", "")
    assert y == "Year"
    assert g == []

def test_normalize_year_equals_fallback():
    y, g = normalize_group_year_cols("Year", "", "Year")
    assert y == "Year"
    assert g == []
