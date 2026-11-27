"""Common-sample four-band coefficients and common-support threshold variants.

Added September 2026 in response to the submission review. Two exact-common-
observation comparisons are frozen here:

1. The four native annual bands (average, median, average_masked,
   median_masked) estimated on identical country-years and identical
   consecutive annual pairs, so that a band difference cannot arise from a
   sample difference.
2. The pixel-threshold variants from ``extract_bgnoise_country.py`` (raw,
   zero-filled, lit-pixel mean, fixed final-year mask) together with the producer
   average and median-masked bands, on one common set of country-years.

Reuses ``build_panel`` and ``BANDS`` from ``analyze_p0_oos.py`` (read-only).
Writes ``results_p0_common_sample.json``, ``data/p0_common_four_band_panel.csv``
and ``data/p0_threshold_common_panel.csv``.
"""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from p0_config import VIIRS_START  # noqa: E402
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import analyze_p0_oos as oos

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def slope(frame: pd.DataFrame, column: str) -> dict:
    """Country-and-year-effects and annual-change slopes, country-clustered."""
    z = frame.copy()
    z["x"] = np.log(z[column])
    z = z.set_index(["country", "year"]).sort_index()
    levels = PanelOLS(
        z["log_y"], z[["x"]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True)
    g = z.reset_index().sort_values(["country", "year"])
    g["dy"] = g.groupby("country")["log_y"].diff()
    g["dx"] = g.groupby("country")["x"].diff()
    g["gap"] = g.groupby("country")["year"].diff()
    g = g[g["gap"].eq(1)].dropna(subset=["dy", "dx"]).set_index(["country", "year"])
    annual = PanelOLS(g["dy"], g[["dx"]], time_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    return {
        "panel_beta": float(levels.params["x"]),
        "panel_se": float(levels.std_errors["x"]),
        "panel_n": int(levels.nobs),
        "annual_beta": float(annual.params["dx"]),
        "annual_se": float(annual.std_errors["dx"]),
        "annual_n": int(annual.nobs),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    panel = oos.build_panel()
    assert panel["year"].min() == VIIRS_START and panel["year"].max() == 2025
    bands = oos.BANDS
    common = panel.dropna(subset=["log_y"] + [f"log_{b}" for b in bands]).copy()
    common.to_csv(DATA / "p0_common_four_band_panel.csv", index=False)
    band_estimates = {band: slope(common, col) for band, col in bands.items()}

    bg = pd.read_csv(DATA / "country_bgnoise_annual.csv")
    threshold = bg.merge(
        panel[["country", "year", "log_y", "avg_mean", "medmask_mean"]],
        on=["country", "year"],
        validate="one_to_one",
    )
    cols = ["raw", "zero", "lit", "stat", "avg_mean", "medmask_mean"]
    threshold = threshold.dropna(subset=cols + ["log_y"])
    threshold = threshold[(threshold[cols] > 0).all(axis=1)].copy()
    threshold.to_csv(DATA / "p0_threshold_common_panel.csv", index=False)
    threshold_estimates = {c: slope(threshold, c) for c in cols}

    result = {
        "common_panel_rows": int(len(common)),
        "common_countries": int(common["country"].nunique()),
        "band_estimates": band_estimates,
        "threshold_rows": int(len(threshold)),
        "threshold_countries": int(threshold["country"].nunique()),
        "threshold_variants": {
            "raw": "cloud-free-coverage-weighted annual mean of all valid pixels",
            "zero": "each year's pixels below 0.4 nW cm-2 sr-1 set to zero",
            "lit": "mean over each year's pixels above 0.4 nW cm-2 sr-1",
            "stat": "mean over pixels above 0.4 nW cm-2 sr-1 in 2025, applied to every year",
            "avg_mean": "producer VNL average band",
            "medmask_mean": "producer VNL median_masked band",
        },
        "threshold_estimates": threshold_estimates,
        "input_sha256": {
            str(p.relative_to(BASE)): sha256(p)
            for p in [
                BASE / "analyze_p0_oos.py",
                DATA / "global_panel.csv",
                DATA / "country_ntl_bands.csv",
                DATA / "country_bgnoise_annual.csv",
            ]
        },
    }
    (BASE / "results_p0_common_sample.json").write_text(json.dumps(result, indent=2))
    show = {k: v for k, v in result.items() if k != "input_sha256"}
    print(json.dumps(show, indent=2))


if __name__ == "__main__":
    main()
