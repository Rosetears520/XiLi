"""
Pytest coverage for refactored xili/rank_tests.py using fixtures.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from xili.rank_tests import run_rank_tests


def test_ranktest_questions_fixture():
    df = pd.read_excel("tests/fixtures/ranktest/ranktest_questions.xlsx")

    question_cols = ["Q1", "Q2"]
    result = run_rank_tests(
        df,
        mode="questions",
        question_cols=question_cols,
        dv_col="BIU",
        alpha=0.05,
    )

    assert list(result.columns) == [
        "Question",
        "TestUsed",
        "Statistic",
        "pValue",
        "Significant",
        "Groups",
    ]
    assert len(result) == len(question_cols)

    result_groups = result.set_index("Question")["Groups"].to_dict()
    for col in question_cols:
        expected_groups = df[col].dropna().nunique()
        assert result_groups[col] == expected_groups


def test_ranktest_group_multi_fixture():
    df = pd.read_excel("tests/fixtures/ranktest/ranktest_group_multi.xlsx")

    group_cols = ["区域", "性别"]
    result = run_rank_tests(
        df,
        mode="group",
        group_cols=group_cols,
        alpha=0.05,
    )

    assert list(result.columns) == ["变量", "检验方法", "statistic", "p值"]

    expected_vars = [col for col in df.columns if col not in group_cols]
    assert sorted(result["变量"].tolist()) == sorted(expected_vars)

    methods = result["检验方法"].astype(str).tolist()
    assert all(not method.startswith("NoTest") for method in methods)
