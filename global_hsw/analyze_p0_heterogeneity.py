"""Where the masking gain arises: held-out masking gains by thirds of countries.

Economies in the held-out sample are split into thirds on characteristics fixed early in the VIIRS
period: unmasked mean radiance in the first three panel years (brightness), the ratio of
masked-average to unmasked-average mean radiance in the same years, GDP per capita in the first
panel year, and the World Bank Statistical Performance Indicators overall score averaged
over 2016-2019 (the first SPI years; first available year otherwise). For each
third, the masking gain is the percentage reduction in summed held-out squared
error of the masked model relative to its unmasked counterpart, using the
predictions written by analyze_p0_oos.py (pooled slopes trained on earlier
endpoints, so the split changes only where errors are evaluated). Intervals
resample countries within each third. The one-year lowest-minus-highest difference is also
recomputed with each evaluation year omitted in turn, keeping the predictions and the thirds
fixed. Annual in-sample coefficients with year
effects and country-clustered standard errors are reported by third on the
common four-band panel.

Inputs:
  data/p0_oos_predictions.csv
  data/p0_common_four_band_panel.csv
  data/wdi_spi_overall.csv  (WDI indicator IQ.SPI.OVRL, vintage 11 November 2026)

Output:
  results_p0_heterogeneity.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p0_config import VIIRS_START  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
REPS = 3999
RNG = np.random.default_rng(20261125)
BANDS = ("average", "median", "average_masked", "median_masked")
LABELS = ("low", "mid", "high")


def characteristics(panel: pd.DataFrame) -> dict[str, pd.Series]:
    early = panel[panel.year.between(VIIRS_START, VIIRS_START + 2)].groupby("iso3")[["avg_mean", "avgmask_mean"]].mean()
    spi = pd.read_csv(DATA / "wdi_spi_overall.csv").dropna()
    first_years = spi[spi.year <= 2019].groupby("iso3").spi.mean()
    fallback = spi.sort_values("year").groupby("iso3").spi.first()
    return {
        "brightness": early.avg_mean,
        "masked_to_unmasked_ratio": early.avgmask_mean / early.avg_mean,
        "gdp_per_capita_initial": panel[panel.year == VIIRS_START].set_index("iso3").gdppc,
        "spi_overall": first_years.combine_first(fallback),
    }


def country_sums(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.assign(sq=(pred.dy_rel - pred.prediction) ** 2, bench=(pred.dy_rel - pred.benchmark) ** 2)
    sums = pred.pivot_table(index="iso3", columns="band", values="sq", aggfunc="sum")[list(BANDS)]
    ref = pred[pred.band == "average"].groupby("iso3")
    sums["benchmark"] = ref.bench.sum()
    sums["n"] = ref.size()
    return sums


def summary(sums: pd.DataFrame) -> dict[str, float]:
    t = sums.sum()
    return {
        "gain_average_pair": 1 - t["average_masked"] / t["average"],
        "gain_median_pair": 1 - t["median_masked"] / t["median"],
        "r2_median_masked": 1 - t["median_masked"] / t["benchmark"],
        "r2_average": 1 - t["average"] / t["benchmark"],
        "rmse_median_masked": float(np.sqrt(t["median_masked"] / t["n"])),
    }


def draws(sums: pd.DataFrame, key: str) -> np.ndarray:
    values = sums.to_numpy()
    cols = list(sums.columns)
    out = np.empty(REPS)
    for r in range(REPS):
        t = values[RNG.integers(0, len(values), len(values))].sum(axis=0)
        s = dict(zip(cols, t))
        out[r] = 1 - s[key.replace("gain_", "").replace("_pair", "") + "_masked"] / s[key.replace("gain_", "").replace("_pair", "")]
    return out


def annual_coefficients(panel: pd.DataFrame, groups: pd.Series) -> dict:
    p = panel.sort_values(["iso3", "year"]).copy()
    p["grp"] = p.iso3.map(groups)
    p["dy"] = p.groupby("iso3").log_y.diff()
    gap = p.groupby("iso3").year.diff()
    for band in ("average", "median_masked"):
        p[f"d_{band}"] = p.groupby("iso3")[f"log_{band}"].diff()
    p = p[gap.eq(1)].dropna(subset=["dy", "grp", "d_average", "d_median_masked"])
    out = {}
    for grp in LABELS:
        x = p[p.grp == grp]
        row = {"n": int(len(x)), "countries": int(x.iso3.nunique())}
        for band in ("average", "median_masked"):
            fit = smf.ols(f"dy ~ d_{band} + C(year)", x).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(x.iso3)[0]})
            row[band] = {"beta": round(float(fit.params[f"d_{band}"]), 6), "se": round(float(fit.bse[f"d_{band}"]), 6)}
        out[grp] = row
    return out


def main() -> None:
    pred_all = pd.read_csv(DATA / "p0_oos_predictions.csv")
    panel = pd.read_csv(DATA / "p0_common_four_band_panel.csv")
    results = {"reps": REPS, "seed": 20261125, "evaluated_economies": int(pred_all.iso3.nunique()), "splits": {}}
    chars = pd.DataFrame(characteristics(panel))
    chars = chars[chars.index.isin(set(pred_all.iso3))]
    results["spearman"] = {f"{a}~{b}": round(float(chars[a].corr(chars[b], method="spearman")), 6)
                           for i, a in enumerate(chars.columns) for b in chars.columns[i + 1:]}
    sample = set(pred_all.iso3)
    # SPI provenance: how many held-out economies have all four 2016-2019 scores, fewer, or only later years
    spi = pd.read_csv(DATA / "wdi_spi_overall.csv").dropna()
    spi = spi[spi.iso3.isin(sample)]
    early_n = spi[spi.year <= 2019].groupby("iso3").year.nunique()
    later_only = sorted(set(spi.iso3) - set(early_n.index))
    results["spi_source_years"] = {"all_four_2016_2019": int((early_n == 4).sum()),
                                   "fewer_than_four_2016_2019": sorted(early_n[early_n < 4].index),
                                   "first_available_year_after_2019": later_only,
                                   "first_year_used": {c: int(spi[spi.iso3 == c].year.min()) for c in later_only},
                                   "no_score": sorted(sample - set(spi.iso3))}
    for name, var in characteristics(panel).items():
        var = var[var.index.isin(sample)]
        groups = pd.qcut(var.dropna(), 3, labels=LABELS)
        split = {"countries_with_value": int(var.notna().sum()),
                 "cutpoints": [round(float(c), 6) for c in var.dropna().quantile([1 / 3, 2 / 3])],
                 "horizons": {}}
        for h in (1, 3, 5):
            sums = country_sums(pred_all[pred_all.horizon == h])
            sums = sums.join(groups.rename("grp"), how="inner")
            block = {}
            for grp in LABELS:
                s = sums[sums.grp == grp].drop(columns="grp")
                row = {"countries": int(len(s)), "n": int(s.n.sum())}
                row.update({k: round(float(v), 6) for k, v in summary(s).items()})
                if h == 1:
                    for key in ("gain_average_pair", "gain_median_pair"):
                        d = draws(s, key)
                        row[f"{key}_ci95"] = [round(float(q), 6) for q in np.percentile(d, [2.5, 97.5])]
                block[grp] = row
            if h == 1:
                lo = sums[sums.grp == "low"].drop(columns="grp")
                hi = sums[sums.grp == "high"].drop(columns="grp")
                for key in ("gain_average_pair", "gain_median_pair"):
                    diff = draws(lo, key) - draws(hi, key)
                    obs = summary(lo)[key] - summary(hi)[key]
                    block[f"low_minus_high_{key}"] = {
                        "estimate": round(float(obs), 6),
                        "ci95": [round(float(q), 6) for q in np.percentile(diff, [2.5, 97.5])]}
                # the same difference with each evaluation year omitted in turn (predictions and thirds fixed)
                p1 = pred_all[pred_all.horizon == 1]
                for key in ("gain_average_pair", "gain_median_pair"):
                    loo = {}
                    for year in sorted(p1.endpoint_year.unique()):
                        t = country_sums(p1[p1.endpoint_year != year]).join(groups.rename("grp"), how="inner")
                        loo[str(int(year))] = round(float(summary(t[t.grp == "low"].drop(columns="grp"))[key]
                                                          - summary(t[t.grp == "high"].drop(columns="grp"))[key]), 6)
                    block[f"low_minus_high_{key}"]["leave_one_year_out"] = loo
            split["horizons"][str(h)] = block
        split["annual_coefficients"] = annual_coefficients(panel, groups)
        results["splits"][name] = split
    with (BASE / "results_p0_heterogeneity.json").open("w") as stream:
        json.dump(results, stream, indent=2)
    for name, split in results["splits"].items():
        b = split["horizons"]["1"]
        print(f"{name}: " + "; ".join(
            f"{g} {100*b[g]['gain_median_pair']:.2f}%" for g in LABELS)
            + f"; low-high {100*b['low_minus_high_gain_median_pair']['estimate']:.2f} pp "
            + str([round(100*x, 2) for x in b['low_minus_high_gain_median_pair']['ci95']]))


if __name__ == "__main__":
    main()
