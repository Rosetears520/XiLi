from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.tools.average_growth_rate import main as run_growth_fill


def test_growth_rate_uses_year_span(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "省份": ["A省", "A省", "A省"],
            "年份": [2020, 2021, 2022],
            "A": [100.0, None, 121.0],
        }
    )
    input_file = tmp_path / "in.csv"
    output_file = tmp_path / "out.csv"
    df.to_csv(input_file, index=False, encoding="utf-8-sig")

    run_growth_fill(
        input_file,
        output_file,
        group_cols=["省份"],
        sort_col="年份",
        cols=["A"],
        id_cols=["省份", "年份"],
        round_values=False,
    )

    out = pd.read_csv(output_file, encoding="utf-8-sig")
    filled = float(out.loc[out["年份"] == 2021, "A"].iloc[0])
    assert abs(filled - 110.0) < 1e-6


def test_growth_rate_backfills_by_reverse(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "省份": ["A省", "A省", "A省"],
            "年份": [2020, 2021, 2022],
            "A": [None, 200.0, 220.0],
        }
    )
    input_file = tmp_path / "in.csv"
    output_file = tmp_path / "out.csv"
    df.to_csv(input_file, index=False, encoding="utf-8-sig")

    run_growth_fill(
        input_file,
        output_file,
        group_cols=["省份"],
        sort_col="年份",
        cols=["A"],
        id_cols=["省份", "年份"],
        round_values=False,
    )

    out = pd.read_csv(output_file, encoding="utf-8-sig")
    filled = float(out.loc[out["年份"] == 2020, "A"].iloc[0])
    assert abs(filled - (200.0 / 1.1)) < 1e-6

