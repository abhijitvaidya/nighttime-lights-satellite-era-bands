"""Two overlap years of DMSP and VIIRS: 2012 and 2013.

Inputs (bundled):
  data/country_vnl2012_masked.csv     EOG masked bands, 2012 composites and 2013, reduced locally
  data/country_vcmcfg_2012_2013.csv   unmasked analogues from monthly composites (Earth Engine)
  data/country_ntl_annual.csv, data/country_ntl_bands.csv   Earth Engine annual bands (2013 level)
  data/country_dmsp_annual.csv        DMSP stable lights
2012 values are placed on the Earth Engine 2013 level: v2012 = EE2013 x source2012 / source2013,
so only each source's own 2012-to-2013 change enters. The main 2012 composite is April 2012-March
2013 (`2012am`, a full year that overlaps calendar 2013 by three months); April-December 2012
(`2012ad`) is reported as a check.

Also re-estimates Leinonen's (2026) specifications over his 2012-2024 years with each 2012 composite.

Outputs: results_p0_overlap_2012.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))
from analyze_p0_vanuatu_sensitivity import panel as build_panel, wdi  # noqa: E402
from country_xwalk import gaul_key  # noqa: E402

BANDS = ["average", "median", "average_masked", "median_masked"]
r4 = lambda x: round(float(x), 6)


def log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def leinonen_2012(Q: pd.DataFrame) -> dict:
    """Leinonen's (2026) Table 1 specifications on the masked average band over his 2012-2024 years:
    country and year effects, country trends, and the long difference between the 2012-13 and 2023-24
    means (economies observed in all four endpoint years)."""
    x = "log_average_masked"
    d = Q[Q.year.between(2012, 2024)].dropna(subset=[x])
    g = pd.factorize(d.iso3)[0]
    fe = smf.ols(f"log_y ~ {x} + C(iso3) + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": g})
    tr = smf.ols(f"log_y ~ {x} + C(iso3) + C(iso3):year + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": g})
    s0, s1 = d[d.year.isin([2012, 2013])], d[d.year.isin([2023, 2024])]
    full = set(s0.groupby("iso3").year.nunique().loc[lambda t: t == 2].index) & set(s1.groupby("iso3").year.nunique().loc[lambda t: t == 2].index)
    a0 = s0[s0.iso3.isin(full)].groupby("iso3")[["log_y", x]].mean()
    a1 = s1[s1.iso3.isin(full)].groupby("iso3")[["log_y", x]].mean()
    ld = smf.ols("dy ~ dx", pd.DataFrame({"dy": a1.log_y - a0.log_y, "dx": a1[x] - a0[x]})).fit(cov_type="HC1")
    r = lambda m, k: [r4(m.params[k]), r4(m.bse[k])]
    return {"country_year_effects": r(fe, x), "country_trends": r(tr, x), "long_difference": r(ld, "dx"),
            "obs": int(len(d)), "economies": int(d.iso3.nunique()), "long_difference_economies": int(len(full))}


def main() -> None:
    loc = pd.read_csv(DATA / "country_vnl2012_masked.csv", dtype={"period": str})
    L = loc.pivot_table(index="gaul_code", columns=["band", "period"], values="mean")
    L.columns = [f"{b}_{p}" for b, p in L.columns]
    M = pd.read_csv(DATA / "country_vcmcfg_2012_2013.csv").set_index("gaul_code")
    a = pd.read_csv(DATA / "country_ntl_annual.csv")
    b = pd.read_csv(DATA / "country_ntl_bands.csv")
    V = (a[a.year == 2013][["gaul_code", "country", "avg_mean", "medmask_mean"]]
         .merge(b[b.year == 2013][["gaul_code", "avgmask_mean", "median_mean"]], on="gaul_code")
         .rename(columns={"avg_mean": "average", "medmask_mean": "median_masked",
                          "avgmask_mean": "average_masked", "median_mean": "median"})
         .set_index("gaul_code"))
    V["iso3"] = gaul_key(V.country)
    panel_codes = set(pd.read_csv(DATA / "global_panel.csv").gaul_code)
    out = {"main_2012_composite": "2012am (April 2012-March 2013)", "check_2012_composite": "2012ad (April-December 2012)"}
    # method check: local 2013 against Earth Engine 2013, panel economies
    cal = {}
    for band in ("average_masked", "median_masked"):
        d = pd.DataFrame({"ee": log(V[band]), "local": log(L[f"{band}_2013"])}).dropna()
        d = d[d.index.isin(panel_codes)]
        g = d.local - d.ee
        cal[band] = {"economies": int(len(d)), "log_corr": r4(d.ee.corr(d.local)), "median_log_gap": r4(g.median()),
                     "sd_log_gap": r4(g.std())}
    out["method_check_2013"] = cal
    for per in ("2012am", "2012ad"):
        for band in ("average_masked", "median_masked"):
            V[f"{band}_{per}"] = V[band] * L[f"{band}_{per}"] / L[f"{band}_2013"]
        V[f"average_{per}"] = V["average"] * M[f"mean_{per}"] / M["mean_2013"]
        V[f"median_{per}"] = V["median"] * M[f"median_{per}"] / M["median_2013"]
    dm = pd.read_csv(DATA / "country_dmsp_annual.csv").pivot_table(index="gaul_code", columns="year", values="stable_mean")
    V["dmsp_2012"], V["dmsp_2013"] = dm[2012], dm[2013]
    # One common sample per 2012 composite: economies in the crosswalk with positive DMSP and positive
    # values of all four bands in both years, so every column of Table O1 uses the same economies.
    w12 = 275 / 365  # days of the April 2012-March 2013 composite in 2012 (90 in 2013)
    V["dmsp_2012am"] = w12 * V.dmsp_2012 + (1 - w12) * V.dmsp_2013

    def compare(per, dmsp12):
        cols = {"dm12": log(V[dmsp12]), "dm13": log(V.dmsp_2013), "iso3": V.iso3}
        for band in BANDS:
            cols[f"v12_{band}"], cols[f"v13_{band}"] = log(V[f"{band}_{per}"]), log(V[band])
        d = pd.DataFrame(cols).dropna()
        res = {"economies": int(len(d))}
        ddm = d.dm13 - d.dm12
        for band in BANDS:
            dv = d[f"v13_{band}"] - d[f"v12_{band}"]
            fit = smf.ols("y ~ x", pd.DataFrame({"y": ddm, "x": dv})).fit(cov_type="HC1")
            res[band] = {"corr_levels_2012": r4(d.dm12.corr(d[f"v12_{band}"])), "corr_levels_2013": r4(d.dm13.corr(d[f"v13_{band}"])),
                         "corr_change": r4(ddm.corr(dv)), "spearman_change": r4(ddm.corr(dv, method="spearman")),
                         "slope_dmsp_change_on_viirs_change": [r4(fit.params["x"]), r4(fit.bse["x"])],
                         "sd_viirs_change": r4(dv.std()), "sd_dmsp_change": r4(ddm.std())}
        return res

    for per in ("2012am", "2012ad"):
        out[f"overlap_{per}"] = compare(per, "dmsp_2012")
    # DMSP has only calendar-year composites; the day-weighted DMSP value approximates the April-March exposure.
    out["overlap_2012am_dmsp_day_weighted"] = {"weights": {"2012": r4(w12), "2013": r4(1 - w12)}, **compare("2012am", "dmsp_2012am")}
    # Main panel (starts in 2014, p0_config.py) with the two uncorrected years added: the 2012 composite
    # and the Earth Engine 2013 values. Leinonen's specifications use a panel that starts in 2013.
    P = build_panel(False)
    P13 = build_panel(False, start=2013)
    gdp = wdi("gdp_constant", "gdp").set_index(["iso3", "year"]).gdp

    def extra_rows(codes, year, values):
        add = V[V.iso3.isin(codes)]
        rows = pd.DataFrame({"iso3": add.iso3.values, "country": add.country.values, "gaul_code": add.index.values, "year": year})
        rows["log_y"] = np.log([gdp.get((c, year), np.nan) for c in rows.iso3])
        for band in BANDS:
            rows[f"log_{band}"] = log(values(add, band)).values
        return rows.dropna(subset=["log_y"])

    for per in ("2012am", "2012ad"):
        r12 = extra_rows(set(P.iso3), 2012, lambda a, b: a[f"{b}_{per}"])
        r13 = extra_rows(set(P.iso3), 2013, lambda a, b: a[b])
        Q = pd.concat([P, r13, r12], ignore_index=True)
        Q13 = pd.concat([P13, extra_rows(set(P13.iso3), 2012, lambda a, b: a[f"{b}_{per}"])], ignore_index=True)
        res = {}
        for band in ("median_masked", "average"):
            x = f"log_{band}"
            d = Q.dropna(subset=[x])
            fe = smf.ols(f"log_y ~ {x} + C(iso3) + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.iso3)[0]})
            q = d.sort_values(["iso3", "year"]).assign(dy=lambda t: t.groupby("iso3").log_y.diff(), dx=lambda t: t.groupby("iso3")[x].diff())
            an = q[q.groupby("iso3").year.diff().eq(1)].dropna(subset=["dy", "dx"])
            af = smf.ols("dy ~ dx + C(year)", an).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(an.iso3)[0]})
            s0, e = d[d.year == 2012].set_index("iso3"), d[d.year == 2025].set_index("iso3")
            c = s0.index.intersection(e.index)
            ld = smf.ols("dy ~ dx", pd.DataFrame({"dy": e.loc[c, "log_y"] - s0.loc[c, "log_y"],
                                                  "dx": e.loc[c, x] - s0.loc[c, x]}).dropna()).fit(cov_type="HC1")
            res[band] = {"country_year_effects": [r4(fe.params[x]), r4(fe.bse[x])],
                         "annual_changes": [r4(af.params["dx"]), r4(af.bse["dx"])],
                         "long_difference_2012_2025": [r4(ld.params["dx"]), r4(ld.bse["dx"])],
                         "obs": int(len(d)), "economies_2012": int((d.year == 2012).sum()), "economies_2013": int((d.year == 2013).sum()), "pairs": int(len(an))}
        out[f"panel_with_{per}"] = res
        out[f"leinonen_specs_2012_2024_{per}"] = leinonen_2012(Q13)
    with (BASE / "results_p0_overlap_2012.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    m = out["overlap_2012am"]
    print("2012am change corr:", {k: m[k]["corr_change"] for k in BANDS}, "economies", m["economies"])
    print("Leinonen specs 2012-2024:", {p: out[f"leinonen_specs_2012_2024_{p}"] for p in ("2012am", "2012ad")})


if __name__ == "__main__":
    main()
