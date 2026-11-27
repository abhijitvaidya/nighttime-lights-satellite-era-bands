"""DMSP validation with satellite-specific data: satellite effects instead of averaging satellites.

HSW, and the main DMSP validation (validate_hsw.py), average the satellites that observe the same
year. DMSP sensors differ in gain and calibration, and Gibson, Olivia and Boe-Gibson (2020)
recommend satellite effects. Using one row per country, satellite and year
(data/country_dmsp_satellite.csv), this script re-estimates the HSW-era (1992-2008) relationships:

  averaged      satellite average of the same extraction (the HSW convention), as a consistency check
  satellite_fe  levels with country, year and satellite effects
  within_sat    annual changes computed within each satellite, with year effects, so no change
                crosses from one sensor to another
  by_satellite  the annual-change coefficient for each satellite separately
  two_satellite_agreement
                correlations between two satellites observing the same country-years: log levels
                (pooled, and within country after removing year means) and annual log changes
                between consecutive years (raw, and after removing each year's mean)

Sample rules follow validate_hsw.py: the crosswalk, WDI GDP of the same vintage, Equatorial Guinea
dropped, positive lights and GDP, and countries with at least 12 years. Standard errors are
clustered by country. Output: results_p0_dmsp_satellite.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))
from country_xwalk import gaul_key  # noqa: E402

r6 = lambda x: round(float(x), 6)


def gdp() -> pd.DataFrame:
    w = pd.read_csv(glob.glob(str(DATA / "gdp_constant/API_*.csv"))[0], skiprows=4)
    yrs = [c for c in w.columns if c.isdigit()]
    g = w.melt(id_vars=["Country Code"], value_vars=yrs, var_name="year", value_name="gdp")
    g["year"] = g.year.astype(int)
    meta = pd.read_csv(glob.glob(str(DATA / "gdp_constant/Metadata_Country_*.csv"))[0])
    return g[g["Country Code"].isin(set(meta.loc[meta.Region.notna(), "Country Code"]))].rename(columns={"Country Code": "iso3"})


def fit(formula: str, d: pd.DataFrame, x: str) -> list:
    m = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.country)[0]})
    return [r6(m.params[x]), r6(m.bse[x])]


def main() -> None:
    s = pd.read_csv(DATA / "country_dmsp_satellite.csv")
    s["iso3"] = gaul_key(s.country)
    s = s.dropna(subset=["iso3"]).merge(gdp(), on=["iso3", "year"])
    s = s[s.year.between(1992, 2008) & (s.country != "Equatorial Guinea") & (s.stable_mean > 0) & (s.gdp > 0)].copy()
    ny = s.groupby("country").year.nunique()
    s = s[s.country.isin(ny[ny >= 12].index)].copy()
    s["log_y"], s["log_l"] = np.log(s.gdp), np.log(s.stable_mean)
    out = {"countries": int(s.country.nunique()), "satellite_year_rows": int(len(s)),
           "satellites": sorted(s.satellite.unique()), "two_satellite_years": sorted(int(y) for y, n in s.groupby("year").satellite.nunique().items() if n > 1)}

    # HSW convention on the same extraction: average the satellites, then log
    a = s.groupby(["country", "year"], as_index=False).agg(stable_mean=("stable_mean", "mean"), log_y=("log_y", "first"))
    a["log_l"] = np.log(a.stable_mean)
    g = a.sort_values(["country", "year"]).assign(dy=lambda t: t.groupby("country").log_y.diff(), dx=lambda t: t.groupby("country").log_l.diff(),
                                                  gap=lambda t: t.groupby("country").year.diff())
    g = g[g.gap == 1].dropna(subset=["dy", "dx"])
    out["averaged"] = {"country_year_effects": fit("log_y ~ log_l + C(country) + C(year)", a, "log_l"),
                       "annual_changes": fit("dy ~ dx + C(year)", g, "dx"), "obs": int(len(a)), "pairs": int(len(g))}

    # Satellite-year rows: levels with satellite effects
    out["satellite_fe"] = {"country_year_effects": fit("log_y ~ log_l + C(country) + C(year) + C(satellite)", s, "log_l"),
                           "obs": int(len(s))}

    # Annual changes within each satellite
    w = s.sort_values(["country", "satellite", "year"]).assign(
        dy=lambda t: t.groupby(["country", "satellite"]).log_y.diff(), dx=lambda t: t.groupby(["country", "satellite"]).log_l.diff(),
        gap=lambda t: t.groupby(["country", "satellite"]).year.diff())
    w = w[w.gap == 1].dropna(subset=["dy", "dx"])
    out["within_sat"] = {"annual_changes": fit("dy ~ dx + C(year)", w, "dx"), "pairs": int(len(w))}
    out["satellite_rows_no_sat_fe"] = {"country_year_effects": fit("log_y ~ log_l + C(country) + C(year)", s, "log_l")}

    # One satellite per year: the newest or the oldest observing satellite
    def one_per_year(newest: bool) -> dict:
        o = s.sort_values(["country", "year", "satellite"], ascending=[True, True, not newest]).drop_duplicates(["country", "year"])
        q = o.sort_values(["country", "year"]).assign(dy=lambda t: t.groupby("country").log_y.diff(), dx=lambda t: t.groupby("country").log_l.diff(),
                                                      gap=lambda t: t.groupby("country").year.diff())
        q = q[q.gap == 1].dropna(subset=["dy", "dx"])
        return {"country_year_effects": fit("log_y ~ log_l + C(country) + C(year)", o, "log_l"), "annual_changes": fit("dy ~ dx + C(year)", q, "dx"),
                "obs": int(len(o)), "pairs": int(len(q))}
    out["newest_satellite"], out["oldest_satellite"] = one_per_year(True), one_per_year(False)

    # Averaged series: annual changes in years with the same satellite set vs years when the set changes
    a2 = s.groupby(["country", "year"]).agg(l=("stable_mean", "mean"), log_y=("log_y", "first"),
                                             sats=("satellite", lambda x: ",".join(sorted(x)))).reset_index()
    a2["log_l"] = np.log(a2.l)
    q = a2.sort_values(["country", "year"]).assign(dy=lambda t: t.groupby("country").log_y.diff(), dx=lambda t: t.groupby("country").log_l.diff(),
                                                   gap=lambda t: t.groupby("country").year.diff(), prev=lambda t: t.groupby("country").sats.shift())
    q = q[q.gap == 1].dropna(subset=["dy", "dx"])
    out["averaged_by_satellite_set"] = {"same_set": fit("dy ~ dx + C(year)", q[q.sats == q.prev], "dx"),
                                        "set_changes": fit("dy ~ dx + C(year)", q[q.sats != q.prev], "dx"),
                                        "set_change_years": sorted(int(y) for y in q.loc[q.sats != q.prev, "year"].unique())}

    # Agreement between two satellites observing the same country-year
    wide = s.pivot_table(index=["country", "year"], columns="satellite", values="log_l")
    agree = {}
    for x1, x2 in (("F10", "F12"), ("F12", "F14"), ("F14", "F15"), ("F15", "F16")):
        if x1 in wide and x2 in wide:
            pair = wide[[x1, x2]].dropna().reset_index().sort_values(["country", "year"])
            by = pair.groupby("country")
            d = by[[x1, x2]].diff()[by.year.diff() == 1].dropna()  # changes between consecutive years only
            dd = d - d.groupby(pair.loc[d.index, "year"]).transform("mean")  # each year's mean change removed
            lw = pair[[x1, x2]] - by[[x1, x2]].transform("mean")
            lw = lw - lw.groupby(pair.year).transform("mean")  # levels within country, year means removed
            ok = len(d) > 10
            agree[f"{x1}_{x2}"] = {"level_corr": r6(pair[x1].corr(pair[x2])),
                                   "level_corr_within_country": r6(lw[x1].corr(lw[x2])) if pair.year.nunique() > 1 else None,
                                   "annual_change_corr": r6(d[x1].corr(d[x2])) if ok else None,
                                   "annual_change_corr_year_demeaned": r6(dd[x1].corr(dd[x2])) if ok else None,
                                   "country_years": int(len(pair)), "changes": int(len(d)),
                                   "change_years": sorted(int(y) for y in pair.loc[d.index, "year"].unique())}
    out["two_satellite_agreement"] = agree
    out["by_satellite"] = {}
    for sat, b in w.groupby("satellite"):
        if b.year.nunique() >= 3:
            out["by_satellite"][sat] = {"annual_changes": fit("dy ~ dx + C(year)", b, "dx"), "years": [int(b.year.min()), int(b.year.max())], "pairs": int(len(b))}

    # Consistency with the main validation (validate_hsw.py, satellite-averaged images)
    main = json.load(open(BASE / "results_hsw_validation.json"))
    out["main_validation"] = {"country_year_effects": [r6(main["levels_2wfe"]["psi"]), r6(main["levels_2wfe"]["se"])],
                              "annual_changes": [r6(main["growth_yearfe"]["psi"]), r6(main["growth_yearfe"]["se"])]}
    with (BASE / "results_p0_dmsp_satellite.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    for k in ("main_validation", "averaged", "satellite_rows_no_sat_fe", "satellite_fe", "within_sat", "newest_satellite", "oldest_satellite", "averaged_by_satellite_set", "two_satellite_agreement"):
        print(k, out[k])
    print("by satellite:", {k: v["annual_changes"] for k, v in out["by_satellite"].items()})


if __name__ == "__main__":
    main()
