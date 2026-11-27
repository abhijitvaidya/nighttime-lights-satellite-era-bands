"""Descriptive statistics for Paper 0's frozen VIIRS country panel.

Inputs:
  data/global_panel.csv

Outputs:
  results_p0_descriptives.json
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from p0_config import VIIRS_START  # noqa: E402
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
INPUT = BASE / "data" / "global_panel.csv"
OUTPUT = BASE / "results_p0_descriptives.json"


def stats(series: pd.Series) -> dict[str, float | int]:
    x = series.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std()),
        "p25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "p75": float(x.quantile(0.75)),
    }


panel = pd.read_csv(INPUT).sort_values(["country", "year"])
panel = panel[(panel["year"] >= VIIRS_START) & (panel["year"] <= 2025)].copy()
panel["gdp_billion_2015_usd"] = panel["gdp"] / 1e9
panel["gdp_growth"] = panel.groupby("country")["log_y"].diff()
panel["masked_radiance_growth"] = panel.groupby("country")["log_l_med"].diff()
consecutive = panel.groupby("country")["year"].diff().eq(1)
panel.loc[~consecutive, ["gdp_growth", "masked_radiance_growth"]] = np.nan

result = {
    "sample": {
        "country_years": int(len(panel)),
        "countries": int(panel["country"].nunique()),
        "first_year": int(panel["year"].min()),
        "last_year": int(panel["year"].max()),
    },
    "variables": {
        "real_gdp_billion_2015_usd": stats(panel["gdp_billion_2015_usd"]),
        "real_gdp_per_capita_2015_usd": stats(panel["gdppc"]),
        "median_masked_mean_radiance": stats(panel["medmask_mean"]),
        "annual_log_real_gdp_change": stats(panel["gdp_growth"]),
        "annual_log_median_masked_radiance_change": stats(
            panel["masked_radiance_growth"]
        ),
    },
}

OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
