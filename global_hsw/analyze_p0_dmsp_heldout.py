"""Held-out prediction on DMSP stable lights, with the VIIRS design of analyze_p0_oos.py.

Same construction as the VIIRS held-out test: for each horizon, GDP and light
changes are demeaned within endpoint year; an expanding-window slope fitted on
earlier endpoints (at least four endpoints, 200 observations, 50 test countries)
predicts each held-out year's relative GDP change; the benchmark predicts zero
relative change. DMSP has one product (stable lights), so there is no masking
comparison; the exercise asks how much relative growth DMSP light changes
predicted, as a reference for the VIIRS bands.

Designs:
  full      all DMSP years, 1992-2013
  hsw_era   1992-2008, the HSW validation window
  windows   every window as long as the main VIIRS panel (p0_config.VIIRS_YEARS years)

Inputs: data/country_dmsp_annual.csv, data/gdp_constant/ (WDI vintage 2026-11-11)
Output: results_p0_dmsp_heldout.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from analyze_p0_oos import horizon_changes, rolling_predictions, summarise  # noqa: E402
from country_xwalk import gaul_key  # noqa: E402
from p0_config import VIIRS_YEARS  # noqa: E402

HORIZONS = (1, 3, 5)
WINDOW = VIIRS_YEARS  # equal to the main VIIRS panel length


def dmsp_panel() -> pd.DataFrame:
    f = glob.glob(str(BASE / "data/gdp_constant/API_*.csv"))[0]
    w = pd.read_csv(f, skiprows=4)
    yrs = [c for c in w.columns if c.isdigit()]
    gdp = w.melt(id_vars=["Country Name", "Country Code"], value_vars=yrs, var_name="year", value_name="gdp")
    gdp["year"] = gdp["year"].astype(int)
    meta = pd.read_csv(glob.glob(str(BASE / "data/gdp_constant/Metadata_Country_*.csv"))[0])
    gdp = gdp[gdp["Country Code"].isin(set(meta[meta["Region"].notna()]["Country Code"]))]
    dmsp = pd.read_csv(BASE / "data/country_dmsp_annual.csv")
    dmsp["iso3"] = gaul_key(dmsp["country"])
    d = dmsp.merge(gdp.rename(columns={"Country Code": "iso3"})[["iso3", "year", "gdp"]], on=["iso3", "year"])
    d = d[d.country.ne("Equatorial Guinea") & (d.stable_mean > 0) & (d.gdp > 0)].copy()
    d["log_y"] = np.log(d.gdp)
    d["log_dmsp"] = np.log(d.stable_mean)
    return d[["country", "iso3", "year", "log_y", "log_dmsp"]].sort_values(["country", "year"])


def run(panel: pd.DataFrame, horizon: int) -> dict | None:
    changes = horizon_changes(panel, "dmsp", horizon)
    pred = rolling_predictions(changes, "dmsp", horizon)
    if pred.empty:
        return None
    s = summarise(pred)
    s.pop("year_results")
    s["countries"] = int(pred.iso3.nunique())
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}


def main() -> None:
    panel = dmsp_panel()
    out = {"panel": {"observations": int(len(panel)), "countries": int(panel.iso3.nunique()),
                     "years": [int(panel.year.min()), int(panel.year.max())]}, "designs": {}}
    for name, (y0, y1) in {"full": (1992, 2013), "hsw_era": (1992, 2008)}.items():
        sub = panel[panel.year.between(y0, y1)]
        out["designs"][name] = {str(h): run(sub, h) for h in HORIZONS}
    windows = {}
    for start in range(1992, 2013 - WINDOW + 2):
        sub = panel[panel.year.between(start, start + WINDOW - 1)]
        windows[f"{start}-{start + WINDOW - 1}"] = {str(h): run(sub, h) for h in HORIZONS}
    out["window_years"] = WINDOW
    out["designs"]["windows_equal_length"] = windows
    summary = {}
    for h in HORIZONS:
        r2 = [w[str(h)]["oos_r2_vs_common_component"] for w in windows.values() if w[str(h)]]
        summary[str(h)] = {"windows": len(r2), "min": round(min(r2), 6), "max": round(max(r2), 6),
                           "mean": round(float(np.mean(r2)), 6), "median": round(float(np.median(r2)), 6)}
    out["windows_equal_length_r2_summary"] = summary
    with (BASE / "results_p0_dmsp_heldout.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["panel"]))
    for name in ("full", "hsw_era"):
        for h in HORIZONS:
            r = out["designs"][name][str(h)]
            print(f"{name:8s} h={h}: R2 {r['oos_r2_vs_common_component']:.3f}  rmse {r['rmse']:.4f} bench {r['benchmark_rmse']:.4f}  "
                  f"test years {r['test_years'][0]}-{r['test_years'][-1]} ({r['n_test_years']})  n {r['n_predictions']}  c {r['countries']}  mean beta {r['mean_training_beta']:.3f}")
    for h in HORIZONS:
        print(f"{WINDOW}-year windows h={h}: {summary[str(h)]}")
    for k, w in windows.items():
        print(k, " ".join(f"h{h}:{w[str(h)]['oos_r2_vs_common_component']:.3f}" if w[str(h)] else f"h{h}:na" for h in HORIZONS))


if __name__ == "__main__":
    main()
