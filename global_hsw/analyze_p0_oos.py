"""Rolling held-out test of the VNL band comparison in Paper 0.

For each one-, three-, and five-year horizon, GDP and light changes are
demeaned within endpoint year. A slope estimated on earlier endpoint years
predicts each held-out year's relative GDP change. The benchmark predicts
zero relative growth, meaning that every country follows the annual common
component.

Inputs:
  data/global_panel.csv
  data/country_ntl_bands.csv

Outputs:
  results_p0_oos.json
  data/p0_oos_predictions.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

BANDS = {
    "average": "avg_mean",
    "median": "median_mean",
    "average_masked": "avgmask_mean",
    "median_masked": "medmask_mean",
}
HORIZONS = (1, 3, 5)
MIN_TRAIN_ENDPOINTS = 4
MIN_TRAIN_OBS = 200
MIN_TEST_COUNTRIES = 50
BOOTSTRAP_REPS = 3999
RNG = np.random.default_rng(20260711)
GAP_RNG = np.random.default_rng(20260712)


def build_panel() -> pd.DataFrame:
    panel = pd.read_csv(DATA / "global_panel.csv")
    extra = pd.read_csv(DATA / "country_ntl_bands.csv")
    panel = panel.merge(
        extra[["gaul_code", "year", "avgmask_mean", "median_mean"]],
        on=["gaul_code", "year"],
        how="left",
        validate="one_to_one",
    )
    for name, col in BANDS.items():
        panel[f"log_{name}"] = np.nan
        valid = panel[col] > 0
        panel.loc[valid, f"log_{name}"] = np.log(panel.loc[valid, col])
    return panel.sort_values(["country", "year"]).reset_index(drop=True)


def horizon_changes(panel: pd.DataFrame, band: str, horizon: int, keep=None) -> pd.DataFrame:
    xcol = f"log_{band}"
    left = panel[["country", "iso3", "year", "log_y", xcol]].copy()
    earlier = left.rename(
        columns={"year": "year0", "log_y": "log_y0", xcol: "log_x0"}
    )
    later = left.rename(
        columns={"year": "year1", "log_y": "log_y1", xcol: "log_x1"}
    )
    d = earlier.merge(later, on=["country", "iso3"], how="inner")
    d = d[d["year1"] - d["year0"] == horizon].copy()
    d["endpoint_year"] = d["year1"]
    d["dy"] = d["log_y1"] - d["log_y0"]
    d["dx"] = d["log_x1"] - d["log_x0"]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["dy", "dx"])
    if keep is not None:
        # Common four-band support: the same country intervals for every band,
        # fixed before the endpoint means are removed and before any slope is fitted.
        d = d.merge(keep, on=["country", "year0", "year1"], how="inner")
    d["dy_rel"] = d["dy"] - d.groupby("endpoint_year")["dy"].transform("mean")
    d["dx_rel"] = d["dx"] - d.groupby("endpoint_year")["dx"].transform("mean")
    return d


def rolling_predictions(changes: pd.DataFrame, band: str, horizon: int,
                        no_overlap: bool = False, min_train: int | None = None) -> pd.DataFrame:
    """Expanding-window predictions. With no_overlap, every training interval must
    end by the start of the evaluated interval, so no annual increment is shared."""
    rows: list[pd.DataFrame] = []
    years = sorted(changes["endpoint_year"].unique())
    for test_year in years:
        cutoff = test_year - horizon if no_overlap else test_year - 1
        prior_years = [year for year in years if year <= cutoff]
        if len(prior_years) < (MIN_TRAIN_ENDPOINTS if min_train is None else min_train):
            continue
        train = changes[changes["endpoint_year"].isin(prior_years)]
        test = changes[changes["endpoint_year"] == test_year].copy()
        if len(train) < MIN_TRAIN_OBS or len(test) < MIN_TEST_COUNTRIES:
            continue
        denominator = float(np.square(train["dx_rel"]).sum())
        if denominator <= 0:
            continue
        beta = float((train["dx_rel"] * train["dy_rel"]).sum() / denominator)
        test["prediction"] = beta * test["dx_rel"]
        test["benchmark"] = 0.0
        test["beta_train"] = beta
        test["band"] = band
        test["horizon"] = horizon
        test["n_train"] = len(train)
        rows.append(
            test[
                [
                    "country", "iso3", "endpoint_year", "horizon", "band",
                    "dy_rel", "dx_rel", "prediction", "benchmark",
                    "beta_train", "n_train",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise(predictions: pd.DataFrame) -> dict:
    errors = predictions["dy_rel"] - predictions["prediction"]
    benchmark_errors = predictions["dy_rel"]
    sse = float(np.square(errors).sum())
    sst = float(np.square(benchmark_errors).sum())
    year_results = []
    for year, group in predictions.groupby("endpoint_year"):
        rho = spearmanr(group["dy_rel"], group["prediction"]).statistic
        year_results.append(
            {
                "year": int(year),
                "n": int(len(group)),
                "rmse": float(np.sqrt(np.mean(np.square(group["dy_rel"] - group["prediction"])))),
                "benchmark_rmse": float(np.sqrt(np.mean(np.square(group["dy_rel"])))),
                "spearman": None if not np.isfinite(rho) else float(rho),
                "beta_train": float(group["beta_train"].iloc[0]),
            }
        )
    rhos = [row["spearman"] for row in year_results if row["spearman"] is not None]
    return {
        "n_predictions": int(len(predictions)),
        "n_test_years": int(predictions["endpoint_year"].nunique()),
        "test_years": [int(year) for year in sorted(predictions["endpoint_year"].unique())],
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "benchmark_rmse": float(np.sqrt(np.mean(np.square(benchmark_errors)))),
        "oos_r2_vs_common_component": None if sst <= 0 else float(1 - sse / sst),
        "median_year_spearman": None if not rhos else float(np.median(rhos)),
        "mean_training_beta": float(
            predictions.groupby("endpoint_year")["beta_train"].first().mean()
        ),
        "year_results": year_results,
    }


def paired_mask_comparison(
    predictions: pd.DataFrame, horizon: int, unmasked: str, masked: str
) -> dict:
    """Country-block bootstrap of held-out squared-error differences."""
    keep = predictions[
        (predictions["horizon"] == horizon)
        & predictions["band"].isin([unmasked, masked])
    ]
    wide = keep.pivot(
        index=["country", "iso3", "endpoint_year"],
        columns="band",
        values=["dy_rel", "prediction"],
    ).dropna()
    actual = wide[("dy_rel", unmasked)]
    difference = np.square(actual - wide[("prediction", unmasked)]) - np.square(
        actual - wide[("prediction", masked)]
    )
    # Country-block bootstrap on the pooled target: resample complete country
    # histories, then take total loss difference over sampled country-endpoint
    # observations, so the draws weight countries as the point estimate does.
    sums_arr = difference.groupby(level="country").sum().to_numpy()
    counts_arr = difference.groupby(level="country").size().to_numpy().astype(float)
    n_countries = len(sums_arr)
    draws = np.empty(BOOTSTRAP_REPS)
    for index in range(BOOTSTRAP_REPS):
        pick = RNG.integers(0, n_countries, size=n_countries)
        draws[index] = sums_arr[pick].sum() / counts_arr[pick].sum()
    estimate = float(difference.mean())
    endpoint_index = difference.index.get_level_values("endpoint_year")
    by_endpoint = difference.groupby(level="endpoint_year").mean()
    leave_one_out = [
        float(difference[endpoint_index != year].mean()) for year in by_endpoint.index
    ]
    return {
        "unmasked": unmasked,
        "masked": masked,
        "mean_squared_error_reduction": estimate,
        "percent_of_unmasked_mse": float(
            estimate
            / np.square(actual - wide[("prediction", unmasked)]).mean()
            * 100
        ),
        "country_block_bootstrap_ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "bootstrap_one_sided_p": float(np.mean(draws <= 0)),
        "bootstrap_draws": int(BOOTSTRAP_REPS),
        "bootstrap_weighting": "pooled country-endpoint target; complete country histories resampled",
        "endpoint_mse_gaps": {int(year): float(value) for year, value in by_endpoint.items()},
        "leave_one_endpoint_out_range": [float(min(leave_one_out)), float(max(leave_one_out))],
        "n_countries": int(n_countries),
        "n_predictions": int(len(difference)),
    }


def weighted_demean(values, groups, weights, n_groups):
    denominators = np.bincount(groups, weights=weights, minlength=n_groups)
    numerators = np.bincount(groups, weights=weights * values, minlength=n_groups)
    means = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=float),
        where=denominators > 0,
    )
    return values - means[groups]


def absorbed_slope(y, x, country_codes, year_codes, weights, two_way):
    """Weighted slope after absorbing country and, optionally, year effects."""
    yr = weighted_demean(y, year_codes, weights, year_codes.max() + 1)
    xr = weighted_demean(x, year_codes, weights, year_codes.max() + 1)
    if two_way:
        yr = y.copy()
        xr = x.copy()
        for _ in range(100):
            old_y = yr.copy()
            old_x = xr.copy()
            yr = weighted_demean(yr, country_codes, weights, country_codes.max() + 1)
            xr = weighted_demean(xr, country_codes, weights, country_codes.max() + 1)
            yr = weighted_demean(yr, year_codes, weights, year_codes.max() + 1)
            xr = weighted_demean(xr, year_codes, weights, year_codes.max() + 1)
            if max(np.max(np.abs(yr - old_y)), np.max(np.abs(xr - old_x))) < 1e-11:
                break
    denominator = float(np.sum(weights * xr * xr))
    return float(np.sum(weights * xr * yr) / denominator)


def paired_coefficient_comparison(panel, unmasked, masked):
    """Country-block bootstrap for masked minus unmasked slopes on the rows passed in."""
    required = ["country", "year", "log_y", f"log_{unmasked}", f"log_{masked}"]
    levels = panel[required].dropna().copy()
    levels["country_code"] = pd.factorize(levels["country"])[0]
    levels["year_code"] = pd.factorize(levels["year"])[0]

    changes = levels.sort_values(["country", "year"]).copy()
    for column in ["log_y", f"log_{unmasked}", f"log_{masked}"]:
        changes[f"d_{column}"] = changes.groupby("country")[column].diff()
    changes["gap"] = changes.groupby("country")["year"].diff()
    changes = changes[changes["gap"] == 1].dropna().copy()
    changes["country_code"] = pd.factorize(changes["country"])[0]
    changes["year_code"] = pd.factorize(changes["year"])[0]

    def arrays(frame, outcome, predictor):
        return (
            frame[outcome].to_numpy(float),
            frame[predictor].to_numpy(float),
            frame["country_code"].to_numpy(int),
            frame["year_code"].to_numpy(int),
        )

    level_y, level_x0, level_c, level_t = arrays(levels, "log_y", f"log_{unmasked}")
    level_x1 = levels[f"log_{masked}"].to_numpy(float)
    growth_y, growth_x0, growth_c, growth_t = arrays(
        changes, "d_log_y", f"d_log_{unmasked}"
    )
    growth_x1 = changes[f"d_log_{masked}"].to_numpy(float)

    countries = sorted(levels["country"].unique())
    country_position = {country: index for index, country in enumerate(countries)}
    level_positions = levels["country"].map(country_position).to_numpy(int)
    growth_positions = changes["country"].map(country_position).to_numpy(int)
    level_weights = np.ones(len(levels))
    growth_weights = np.ones(len(changes))

    observed = {
        "levels_2wfe": absorbed_slope(level_y, level_x1, level_c, level_t, level_weights, True)
        - absorbed_slope(level_y, level_x0, level_c, level_t, level_weights, True),
        "growth_yearfe": absorbed_slope(growth_y, growth_x1, growth_c, growth_t, growth_weights, False)
        - absorbed_slope(growth_y, growth_x0, growth_c, growth_t, growth_weights, False),
    }
    draws = {key: np.empty(BOOTSTRAP_REPS) for key in observed}
    for index in range(BOOTSTRAP_REPS):
        counts = np.bincount(
            GAP_RNG.integers(0, len(countries), size=len(countries)),
            minlength=len(countries),
        )
        lw = counts[level_positions].astype(float)
        gw = counts[growth_positions].astype(float)
        draws["levels_2wfe"][index] = absorbed_slope(
            level_y, level_x1, level_c, level_t, lw, True
        ) - absorbed_slope(level_y, level_x0, level_c, level_t, lw, True)
        draws["growth_yearfe"][index] = absorbed_slope(
            growth_y, growth_x1, growth_c, growth_t, gw, False
        ) - absorbed_slope(growth_y, growth_x0, growth_c, growth_t, gw, False)

    output = {
        "unmasked": unmasked,
        "masked": masked,
        "n_countries": len(countries),
        "n_level_observations": len(levels),
        "n_growth_observations": len(changes),
    }
    for specification, estimate in observed.items():
        distribution = draws[specification]
        output[specification] = {
            "masked_minus_unmasked": estimate,
            "country_block_bootstrap_ci95": [
                float(np.percentile(distribution, 2.5)),
                float(np.percentile(distribution, 97.5)),
            ],
            "bootstrap_two_sided_p": float(
                min(
                    1.0,
                    2
                    * (min(np.sum(distribution <= 0), np.sum(distribution >= 0)) + 1)
                    / (len(distribution) + 1),
                )
            ),
        }
    return output



def forecast_tests(predictions: pd.DataFrame) -> dict:
    """Formal comparisons of held-out predictions (Clark and McCracken 2013).

    Clark-West (2007), each band against the common-component benchmark (nested: the benchmark
    predicts zero relative change). With a zero benchmark the adjusted loss differential is
    dy^2 - [(dy - p)^2 - p^2] = 2 dy p; the test is one-sided (model better).
    Masked against unmasked (non-nested; Diebold-Mariano type): d = e_unmasked^2 - e_masked^2 on
    matched country-endpoints. Both use the mean of the differential with country-clustered
    standard errors, which absorb serial correlation within a country, including the overlap of
    multi-year intervals; they do not model dependence between countries within an endpoint.
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    def clustered_mean(values: pd.Series, groups: pd.Series) -> dict:
        fit = sm.OLS(values.to_numpy(), np.ones(len(values))).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
        t = float(fit.params[0] / fit.bse[0])
        return {"mean": float(fit.params[0]), "se": float(fit.bse[0]), "t": t,
                "p_one_sided": float(1 - norm.cdf(t)), "p_two_sided": float(2 * (1 - norm.cdf(abs(t)))),
                "n": int(len(values)), "clusters": int(groups.nunique())}

    out = {"clark_west": {}, "masked_vs_unmasked": {}}
    for horizon, frame in predictions.groupby("horizon"):
        h = str(int(horizon))
        out["clark_west"][h] = {}
        for band, b in frame.groupby("band"):
            out["clark_west"][h][band] = clustered_mean(2 * b.dy_rel * b.prediction, b.iso3)
        out["masked_vs_unmasked"][h] = {}
        keys = ["iso3", "endpoint_year"]
        for pair, (unmasked, masked) in {"mean": ("average", "average_masked"),
                                         "median": ("median", "median_masked")}.items():
            u = frame[frame.band == unmasked].set_index(keys)
            m = frame[frame.band == masked].set_index(keys)
            j = u[["dy_rel", "prediction"]].join(m[["prediction"]], rsuffix="_m", how="inner").reset_index()
            d = (j.dy_rel - j.prediction) ** 2 - (j.dy_rel - j.prediction_m) ** 2
            out["masked_vs_unmasked"][h][pair] = clustered_mean(d, j.iso3)
    return out


def min_training_sensitivity(panel: pd.DataFrame, minimums=(3, 5)) -> dict:
    """Held-out R-squared and masking gains when the minimum number of earlier training endpoints
    is 3 or 5 instead of 4. No bootstrap, so the shared random-number stream is not touched."""
    out = {}
    for m in minimums:
        out[str(m)] = {}
        for horizon in HORIZONS:
            raw = {band: horizon_changes(panel, band, horizon) for band in BANDS}
            keep = None
            for frame in raw.values():
                keys = frame[["country", "year0", "year1"]]
                keep = keys if keep is None else keep.merge(keys, on=["country", "year0", "year1"])
            preds = pd.concat([rolling_predictions(horizon_changes(panel, b, horizon, keep), b, horizon, min_train=m) for b in BANDS])
            if preds.empty:
                continue
            sse = preds.assign(e=(preds.dy_rel - preds.prediction) ** 2).groupby("band").e.sum()
            sst = float((preds[preds.band == "average"].dy_rel ** 2).sum())
            years = sorted(preds.endpoint_year.unique())
            out[str(m)][str(horizon)] = {
                "endpoints": [int(years[0]), int(years[-1])], "n_endpoints": len(years),
                "r2": {b: float(1 - sse[b] / sst) for b in BANDS},
                "gain_average_pair_pct": float(100 * (1 - sse["average_masked"] / sse["average"])),
                "gain_median_pair_pct": float(100 * (1 - sse["median_masked"] / sse["median"]))}
    return out

def main() -> None:
    panel = build_panel()
    all_predictions = []
    results = {
        "design": {
            "target": "within-endpoint-year relative log GDP change",
            "predictor": "within-endpoint-year relative log light change",
            "benchmark": "annual common component only (zero relative growth)",
            "minimum_training_endpoint_years": MIN_TRAIN_ENDPOINTS,
            "minimum_test_countries": MIN_TEST_COUNTRIES,
        },
        "results": {},
    }
    results["design"]["common_support"] = ("country intervals with all four bands, "
                                           "intersected at each horizon before demeaning and fitting")
    results["no_overlap"] = {}
    no_overlap_predictions = []
    for horizon in HORIZONS:
        raw = {band: horizon_changes(panel, band, horizon) for band in BANDS}
        keep = None
        for frame in raw.values():
            keys = frame[["country", "year0", "year1"]]
            keep = keys if keep is None else keep.merge(keys, on=["country", "year0", "year1"])
        results["results"][str(horizon)] = {}
        results["no_overlap"][str(horizon)] = {}
        for band in BANDS:
            changes = horizon_changes(panel, band, horizon, keep)
            predictions = rolling_predictions(changes, band, horizon)
            if not predictions.empty:
                all_predictions.append(predictions)
                results["results"][str(horizon)][band] = summarise(predictions)
            strict = rolling_predictions(changes, band, horizon, no_overlap=True)
            if not strict.empty:
                no_overlap_predictions.append(strict)
                results["no_overlap"][str(horizon)][band] = summarise(strict)

    predictions = pd.concat(all_predictions, ignore_index=True)
    results["paired_mask_comparisons"] = {}
    for horizon in HORIZONS:
        results["paired_mask_comparisons"][str(horizon)] = {
            "mean": paired_mask_comparison(
                predictions, horizon, "average", "average_masked"
            ),
            "median": paired_mask_comparison(
                predictions, horizon, "median", "median_masked"
            ),
        }
    strict_all = pd.concat(no_overlap_predictions, ignore_index=True)
    results["no_overlap_paired_mask_comparisons"] = {}
    for horizon in HORIZONS:
        if strict_all[strict_all["horizon"] == horizon].empty:
            results["no_overlap_paired_mask_comparisons"][str(horizon)] = None
            continue
        results["no_overlap_paired_mask_comparisons"][str(horizon)] = {
            "mean": paired_mask_comparison(strict_all, horizon, "average", "average_masked"),
            "median": paired_mask_comparison(strict_all, horizon, "median", "median_masked"),
        }
    # Coefficient gaps use the four-band common sample, the same rows as the band
    # comparison in the paper's Table 3 (results_p0_common_sample.json).
    four_band = panel.dropna(subset=["log_y"] + [f"log_{band}" for band in BANDS])
    results["paired_coefficient_comparisons"] = {
        "mean": paired_coefficient_comparison(four_band, "average", "average_masked"),
        "median": paired_coefficient_comparison(four_band, "median", "median_masked"),
    }
    results["forecast_tests"] = forecast_tests(predictions)
    results["min_training_sensitivity"] = min_training_sensitivity(panel)
    predictions.to_csv(DATA / "p0_oos_predictions.csv", index=False)
    with (BASE / "results_p0_oos.json").open("w") as stream:
        json.dump(results, stream, indent=2)

    print("Rolling held-out relative-growth prediction")
    for horizon, bands in results["results"].items():
        print(f"\nHorizon: {horizon} year(s)")
        for band, result in bands.items():
            print(
                f"  {band:<16} RMSE={result['rmse']:.4f} "
                f"baseline={result['benchmark_rmse']:.4f} "
                f"OOS R2={result['oos_r2_vs_common_component']:+.3f} "
                f"median rho={result['median_year_spearman']:+.3f} "
                f"n={result['n_predictions']}"
            )
    print("\nwritten: results_p0_oos.json, data/p0_oos_predictions.csv")


if __name__ == "__main__":
    main()
