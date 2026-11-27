"""Whole-history reassignment on fixed support (synthetic matching diagnostic).

Replaces the exploratory 200-draw block permutation in analyze_global_pilot.py,
which allowed self-matches and inner-joined donor histories onto recipient rows
(dropping rows when a donor lacked a year). Here complete radiance histories are
reassigned only among countries with identical observed-year patterns, so every
recipient keeps its rows and missingness; strata with one country are excluded
from both the observed and reassigned statistics; 1,999 draws.

Statistics recomputed on the eligible sample in every draw:
  * country-effects level coefficient (log GDP on log radiance, within country)
  * annual-change coefficient with year effects.
Writes results_p0_permutation.json and data/p0_permutation_draws.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_p0_oos as oos

BASE = Path(__file__).resolve().parent
DRAWS = 1999
SEED = 20260926


def run(panel: pd.DataFrame, band: str) -> tuple[dict, np.ndarray]:
    d = panel.dropna(subset=["log_y", f"log_{band}"])[["country", "year", "log_y", f"log_{band}"]].copy()
    d = d.rename(columns={f"log_{band}": "log_l"})
    hist = d.groupby("country").year.apply(tuple)
    strata = {}
    for c, h in hist.items():
        strata.setdefault(h, []).append(c)
    strata = [v for v in strata.values() if len(v) >= 2]
    keep = sorted(c for g in strata for c in g)
    d = d[d.country.isin(keep)]
    y = d.pivot(index="country", columns="year", values="log_y").reindex(keep).to_numpy()
    x = d.pivot(index="country", columns="year", values="log_l").reindex(keep).to_numpy()
    pos = {c: i for i, c in enumerate(keep)}
    blocks = [np.array([pos[c] for c in g]) for g in strata]

    def slope(a, b):
        v = np.isfinite(a) & np.isfinite(b)
        return float(np.sum(a[v] * b[v]) / np.sum(b[v] ** 2))

    def stats(xx):
        fe = slope(y - np.nanmean(y, axis=1)[:, None], xx - np.nanmean(xx, axis=1)[:, None])
        dy, dx = np.diff(y, axis=1), np.diff(xx, axis=1)
        annual = slope(dy - np.nanmean(dy, axis=0), dx - np.nanmean(dx, axis=0))
        return [fe, annual]

    observed = stats(x)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(DRAWS):
        ix = np.arange(len(x))
        for b in blocks:
            ix[b] = rng.permutation(b)
        xx = x[ix]
        assert np.array_equal(np.isnan(x), np.isnan(xx))
        out.append(stats(xx))
    arr = np.array(out)
    present = np.isfinite(y)
    res = dict(band=band, countries=len(keep), strata=len(strata), level_rows=int(present.sum()),
               annual_pairs=int((present[:, 1:] & present[:, :-1]).sum()), draws=DRAWS, seed=SEED,
               excluded_singleton_countries=int(hist.size - len(keep)),
               statistics={lab: dict(observed=observed[j], mean=float(arr[:, j].mean()),
                                     range95=np.quantile(arr[:, j], [.025, .975]).tolist(),
                                     share_draws_at_or_above_observed=float(np.mean(arr[:, j] >= observed[j])))
                           for j, lab in enumerate(["country_effects", "annual_year_effects"])})
    return res, arr


def main():
    panel = oos.build_panel()
    results, frames = {}, []
    for band in ["median_masked", "average"]:
        res, arr = run(panel, band)
        results[band] = res
        frames.append(pd.DataFrame(arr, columns=["country_effects", "annual_year_effects"]).assign(band=band))
    (BASE / "results_p0_permutation.json").write_text(json.dumps(results, indent=2))
    pd.concat(frames).to_csv(BASE / "data" / "p0_permutation_draws.csv", index=False)
    print(json.dumps({b: {k: v for k, v in r.items() if k != "statistics"} | {s: {kk: round(vv, 4) if isinstance(vv, float) else [round(q, 4) for q in vv] for kk, vv in st.items()} for s, st in r["statistics"].items()} for b, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
