"""Main VIIRS estimates with and without Vanuatu, plus influence checks.

Vanuatu is excluded in the crosswalk (data/gaul_country_iso3.csv) because the lava
lakes of the Ambrym volcano dominated its radiance until they drained in December
2018 (Shreve et al. 2019). This script rebuilds the VIIRS panel with Vanuatu added
back, using the same sample rules as analyze_global_pilot.py, and reports the main
quantities both ways. It also reports leave-one-economy-out ranges after the
exclusion, an estimate without economies of fewer than one million people, and
Leinonen's (2026) three HSW specifications re-estimated on the masked average band
for 2013-2024, on a panel that starts in 2013 to match his window (the main panel starts in
2014; p0_config.py). His long difference compares the 2012-13 and 2023-24 means; ours
starts from 2013 alone (no 2012 composite in Earth Engine) and ends at each
economy's mean over available 2023-24 values.

Inputs: data/country_ntl_annual.csv, data/country_ntl_bands.csv, data/gdp_constant/,
        data/gdppc_constant/, data/gaul_country_iso3.csv
Output: results_p0_vanuatu_sensitivity.json
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
from analyze_p0_oos import BANDS, horizon_changes, rolling_predictions  # noqa: E402
from p0_config import VIIRS_START  # noqa: E402


def wdi(folder: str, name: str) -> pd.DataFrame:
    w = pd.read_csv(glob.glob(str(DATA / folder / "API_*.csv"))[0], skiprows=4)
    yrs = [c for c in w.columns if c.isdigit()]
    out = w.melt(id_vars=["Country Code"], value_vars=yrs, var_name="year", value_name=name)
    out["year"] = out.year.astype(int)
    return out.rename(columns={"Country Code": "iso3"})


def panel(include_vanuatu: bool, start: int = VIIRS_START) -> pd.DataFrame:
    xw = pd.read_csv(DATA / "gaul_country_iso3.csv", keep_default_na=False)
    keep = xw.status.eq("included") | (include_vanuatu & xw.country.eq("Vanuatu"))
    iso = dict(xw.loc[keep, ["country", "iso3"]].values)
    meta = pd.read_csv(glob.glob(str(DATA / "gdp_constant/Metadata_Country_*.csv"))[0])
    real = set(meta.loc[meta.Region.notna(), "Country Code"])
    g = wdi("gdp_constant", "gdp").merge(wdi("gdppc_constant", "gdppc"), on=["iso3", "year"])
    g = g[g.iso3.isin(real)]
    ntl = pd.read_csv(DATA / "country_ntl_annual.csv")
    ntl = ntl[ntl.year >= start]
    ntl["iso3"] = ntl.country.map(iso)
    p = ntl.dropna(subset=["iso3"]).merge(g, on=["iso3", "year"]).dropna(subset=["gdp"])
    p = p[(p.avg_mean > 0) & (p.gdp > 0)].copy()
    ny = p.groupby("country").year.nunique()
    p = p[p.country.isin(ny[ny >= 8].index)].copy()
    extra = pd.read_csv(DATA / "country_ntl_bands.csv")[["gaul_code", "year", "avgmask_mean", "median_mean"]]
    p = p.merge(extra, on=["gaul_code", "year"], how="left", validate="one_to_one")
    p["log_y"] = np.log(p.gdp)
    for name, col in {"average": "avg_mean", "median": "median_mean", "average_masked": "avgmask_mean", "median_masked": "medmask_mean"}.items():
        p[f"log_{name}"] = np.where(p[col] > 0, np.log(p[col].where(p[col] > 0)), np.nan)
    return p.sort_values(["country", "year"]).reset_index(drop=True)


def ladder(p: pd.DataFrame, band: str) -> dict:
    x = f"log_{band}"
    d = p.dropna(subset=[x])
    fe = smf.ols(f"log_y ~ {x} + C(iso3) + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.iso3)[0]})
    d = d.sort_values(["iso3", "year"]).assign(dy=lambda t: t.groupby("iso3").log_y.diff(), dx=lambda t: t.groupby("iso3")[x].diff())
    a = d[d.groupby("iso3").year.diff().eq(1)].dropna(subset=["dy", "dx"])
    an = smf.ols("dy ~ dx + C(year)", a).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(a.iso3)[0]})
    s, e = d[d.year == d.year.min()].set_index("iso3"), d[d.year == 2025].set_index("iso3")
    c = s.index.intersection(e.index)
    L = pd.DataFrame({"dy": e.loc[c, "log_y"] - s.loc[c, "log_y"], "dx": e.loc[c, x] - s.loc[c, x]}).dropna()
    ld = smf.ols("dy ~ dx", L).fit(cov_type="HC1")
    return {"fe": (fe.params[x], fe.bse[x]), "annual": (an.params["dx"], an.bse["dx"]), "ld": (ld.params["dx"], ld.bse["dx"]),
            "countries": int(d.iso3.nunique()), "obs": int(len(d))}


def heldout(p: pd.DataFrame) -> dict:
    out = {}
    for h in (1, 3, 5):
        raw = {b: horizon_changes(p, b, h) for b in BANDS}
        keep = None
        for d in raw.values():
            k = d[["country", "year0", "year1"]].drop_duplicates()
            keep = k if keep is None else keep.merge(k, on=["country", "year0", "year1"])
        pr = pd.concat([rolling_predictions(horizon_changes(p, b, h, keep), b, h) for b in BANDS])
        sse = pr.assign(e=(pr.dy_rel - pr.prediction) ** 2).groupby("band").e.sum()
        sst = float((pr[pr.band == "average"].dy_rel ** 2).sum())
        out[h] = {"r2_median_masked": 1 - sse["median_masked"] / sst, "r2_average": 1 - sse["average"] / sst,
                  "gain_average": 1 - sse["average_masked"] / sse["average"], "gain_median": 1 - sse["median_masked"] / sse["median"]}
    return out


def twfe(d: pd.DataFrame, x: str) -> float:
    d = d.dropna(subset=["log_y", x])[["iso3", "year", "log_y", x]].copy()
    for v in ("log_y", x):
        for _ in range(40):
            d[v] -= d.groupby("iso3")[v].transform("mean")
            d[v] -= d.groupby("year")[v].transform("mean")
    return float((d[x] * d.log_y).sum() / (d[x] ** 2).sum())


def annual(d: pd.DataFrame, x: str) -> float:
    d = d.sort_values(["iso3", "year"]).copy()
    d["dy"], d["dx"] = d.groupby("iso3").log_y.diff(), d.groupby("iso3")[x].diff()
    d = d[d.groupby("iso3").year.diff().eq(1)].dropna(subset=["dy", "dx"])
    for v in ("dy", "dx"):
        d[v] -= d.groupby("year")[v].transform("mean")
    return float((d.dx * d.dy).sum() / (d.dx ** 2).sum())


def leinonen_specs(p: pd.DataFrame) -> dict:
    """Leinonen's (2026) Table 1 specifications on the masked average band, 2013-2024."""
    x = "log_average_masked"
    d = p[p.year.between(2013, 2024)].dropna(subset=[x])
    g = pd.factorize(d.iso3)[0]
    fe = smf.ols(f"log_y ~ {x} + C(iso3) + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": g})
    tr = smf.ols(f"log_y ~ {x} + C(iso3) + C(iso3):year + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": g})
    q = d.sort_values(["iso3", "year"]).assign(dy=lambda t: t.groupby("iso3").log_y.diff(), dx=lambda t: t.groupby("iso3")[x].diff())
    a = q[q.groupby("iso3").year.diff().eq(1)].dropna(subset=["dy", "dx"])
    an = smf.ols("dy ~ dx + C(year)", a).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(a.iso3)[0]})
    s0 = d[d.year == 2013].groupby("iso3")[["log_y", x]].mean()
    s1 = d[d.year.isin([2023, 2024])].groupby("iso3")[["log_y", x]].mean()
    c = s0.index.intersection(s1.index)
    L = pd.DataFrame({"dy": s1.loc[c, "log_y"] - s0.loc[c, "log_y"], "dx": s1.loc[c, x] - s0.loc[c, x]})
    ld = smf.ols("dy ~ dx", L).fit(cov_type="HC1")
    r = lambda m, k: [round(float(m.params[k]), 6), round(float(m.bse[k]), 6)]
    return {"country_year_effects": r(fe, x), "country_trends": r(tr, x), "long_difference": r(ld, "dx"),
            "annual_changes": r(an, "dx"), "obs": int(len(d)), "economies": int(d.iso3.nunique()), "long_difference_economies": int(len(L))}



def gbg_window(p: pd.DataFrame) -> dict:
    """Country and year effects on the 2014-2019 window of Gibson and Boe-Gibson (2021, Table 1)."""
    d0 = p[p.year.between(2014, 2019)]
    out = {}
    for band in ("average", "average_masked", "median_masked"):
        x = f"log_{band}"
        d = d0.dropna(subset=[x])
        fe = smf.ols(f"log_y ~ {x} + C(iso3) + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.iso3)[0]})
        out[band] = [round(float(fe.params[x]), 6), round(float(fe.bse[x]), 6)]
    out["obs"], out["economies"] = int(len(d0)), int(d0.iso3.nunique())
    return out

def main() -> None:
    inc, exc = panel(True), panel(False)
    assert len(exc) == len(pd.read_csv(DATA / "global_panel.csv")), "rebuilt panel differs from global_panel.csv"
    res = {"excluded": {}, "included": {}}
    fmt = lambda v: f"{v[0]:.3f} ({v[1]:.3f})".replace("-", "−")
    pct = lambda v: f"{100 * v:.2f}%".replace("-", "−")
    for label, p in (("excluded", exc), ("included", inc)):
        mm, av, ho = ladder(p, "median_masked"), ladder(p, "average"), heldout(p)
        res[label] = {
            "sample": f"{mm['countries']} / {mm['obs']:,}",
            "mm_fe": fmt(mm["fe"]), "mm_annual": fmt(mm["annual"]), "mm_ld": fmt(mm["ld"]),
            "av_fe": fmt(av["fe"]), "av_annual": fmt(av["annual"]), "av_ld": fmt(av["ld"]),
            "r2_1": f"{ho[1]['r2_median_masked']:.3f}", "r2_3": f"{ho[3]['r2_median_masked']:.3f}", "r2_5": f"{ho[5]['r2_median_masked']:.3f}",
            "gain_1": f"{pct(ho[1]['gain_average'])} / {pct(ho[1]['gain_median'])}",
            "gain_3": f"{pct(ho[3]['gain_average'])} / {pct(ho[3]['gain_median'])}",
            "gain_5": f"{pct(ho[5]['gain_average'])} / {pct(ho[5]['gain_median'])}",
        }
    res["rows"] = [
        ("Economies / country-years", "sample"),
        ("Median-masked, country and year effects", "mm_fe"), ("Median-masked, annual changes", "mm_annual"),
        (f"Median-masked, long difference {VIIRS_START}–2025", "mm_ld"),
        ("Unmasked average, country and year effects", "av_fe"), ("Unmasked average, annual changes", "av_annual"),
        (f"Unmasked average, long difference {VIIRS_START}–2025", "av_ld"),
        ("Held-out $R^2$, median-masked, one year", "r2_1"), ("Held-out $R^2$, median-masked, three years", "r2_3"),
        ("Held-out $R^2$, median-masked, five years", "r2_5"),
        ("Masking gain (MSE), average / median pair, one year", "gain_1"),
        ("Masking gain (MSE), average / median pair, three years", "gain_3"),
        ("Masking gain (MSE), average / median pair, five years", "gain_5"),
    ]
    v = inc[inc.iso3 == "VUT"].set_index("year").log_median_masked
    res["vanuatu_log_radiance"] = {str(k): round(float(x), 4) for k, x in v.items()}
    res["vanuatu_fall_pct"] = round(float((1 - np.exp(v[2019] - v[2018])) * 100), 1)
    x = "log_median_masked"
    b_fe, b_an = twfe(inc, x), annual(inc, x)
    shifts = [(c, twfe(inc[inc.iso3 != c], x) - b_fe, annual(inc[inc.iso3 != c], x) - b_an) for c in sorted(inc.iso3.unique())]
    s = pd.DataFrame(shifts, columns=["iso3", "fe", "an"]).set_index("iso3")
    nxt_fe = s.drop("VUT").fe.abs().idxmax(); nxt_an = s.drop("VUT").an.abs().idxmax()
    res["loo"] = {"vanuatu_shift_levels": round(float(s.loc["VUT", "fe"]), 4), "vanuatu_shift_annual": round(float(s.loc["VUT", "an"]), 4),
                  "next_levels": f"{s.loc[nxt_fe, 'fe']:+.3f} ({nxt_fe})", "next_annual": f"{s.loc[nxt_an, 'an']:+.3f} ({nxt_an})"}
    a_fe, a_an = twfe(exc, x), annual(exc, x)
    after = [(twfe(exc[exc.iso3 != c], x), annual(exc[exc.iso3 != c], x)) for c in sorted(exc.iso3.unique())]
    res["loo_after"] = {"levels_min": round(min(t[0] for t in after), 4), "levels_max": round(max(t[0] for t in after), 4),
                        "annual_min": round(min(t[1] for t in after), 4), "annual_max": round(max(t[1] for t in after), 4),
                        "levels_full": round(a_fe, 4), "annual_full": round(a_an, 4)}
    pop = (exc.gdp / exc.gdppc).groupby(exc.iso3).mean()
    big = exc[exc.iso3.isin(pop[pop >= 1e6].index)]
    res["small"] = {"n": int((pop < 1e6).sum()), "levels": round(twfe(big, x), 4), "annual": round(annual(big, x), 4)}
    res["gibson_boe_gibson_window"] = {"vanuatu_excluded": gbg_window(exc), "vanuatu_included": gbg_window(inc),
                                       "their_table1": {"average": [0.015, 0.007], "average_masked": [0.094, 0.038], "obs": 1192, "economies": 203}}
    res["leinonen_comparison"] = {
        "leinonen_2026": {"country_year_effects": [0.225, 0.033], "country_trends": [0.124, 0.022],
                          "long_difference": [0.270, 0.042], "annual_changes": None, "obs": 2509, "economies": 195,
                          "long_difference_economies": 192, "source": "Leinonen (2026), Table 1; transcribed"},
        "vanuatu_included": leinonen_specs(panel(True, start=2013)), "vanuatu_excluded": leinonen_specs(panel(False, start=2013))}  # his window includes 2013
    with (BASE / "results_p0_vanuatu_sensitivity.json").open("w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for lab, key in res["rows"]:
        print(f"{lab:58s} {res['excluded'][key]:>22s} {res['included'][key]:>22s}")
    print(res["loo"], res["loo_after"], res["small"], res["vanuatu_fall_pct"])


if __name__ == "__main__":
    main()
