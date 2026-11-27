"""Calendar-year lights against fiscal-year GDP: a timing check for the VIIRS panel.

The annual VNL composites cover January-December. World Development Indicators report national
accounts on a fiscal-year basis for some economies (WDI country metadata, `SpecialNotes`: "fiscal
year-based (fiscal year-end: ...)" or "reporting period for national accounts data: FY"). WDI
assigns a fiscal year to the calendar year containing the larger share of it: a fiscal year ending
before 30 June is shown in its first calendar year, one ending on or after 30 June in its second
(World Bank Data Help Desk, "Are all WDI series on a calendar year reporting period?"). India's 2020
is April 2020 to March 2021 and Pakistan's 2020 is July 2019 to June 2020; the 2020 GDP falls in
the bundled data agree. The metadata give one year-end per economy; Myanmar, whose fiscal year
changed during the panel, is left out of the day-weighted version.

Three versions of the main sample (Vanuatu excluded) are compared:
  calendar   the paper's panel: calendar-year lights for every economy;
  excluded   fiscal-year reporters dropped;
  aligned    for fiscal-year reporters except Myanmar, each band's radiance for fiscal year y is
             approximated by the day-weighted mean of the two calendar-year composites it spans,
             then logged; observations whose fiscal year reaches outside the panel years are dropped.
Masked bands exist only as annual composites, so a fiscal-year composite cannot be built from
monthly data for them; day weighting treats all four bands alike.

Inputs: as analyze_p0_vanuatu_sensitivity.py, plus the WDI country metadata.
Output: results_p0_fiscal_year.json
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))
from analyze_p0_vanuatu_sensitivity import heldout, ladder, panel  # noqa: E402
from p0_config import VIIRS_START  # noqa: E402

COLS = {"average": "avg_mean", "median_masked": "medmask_mean", "average_masked": "avgmask_mean", "median": "median_mean"}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
r6 = lambda x: round(float(x), 6)


def fiscal_reporters() -> dict:
    """ISO3 -> (end month, end day) for economies whose national accounts are fiscal-year based."""
    meta = pd.read_csv(glob.glob(str(DATA / "gdp_constant/Metadata_Country_*.csv"))[0]).fillna("")
    out = {}
    for _, r in meta.iterrows():
        n = r.SpecialNotes
        if not re.search(r"fiscal year-based|national accounts data: FY", n, re.I):
            continue
        e = re.search(r"fiscal year[- ]end:?\s*(\w+) (\d+)|fiscal year ends on (\w+) (\d+)", n, re.I)
        mon, day = (e.group(1), e.group(2)) if e.group(1) else (e.group(3), e.group(4))
        out[r["Country Code"]] = (MONTHS[mon.lower()], int(day))
    return out


def weights(end: tuple, label: int) -> dict:
    """Calendar year -> share of days of the fiscal year WDI labels `label`."""
    m, d = end
    if (m, d) >= (6, 30):  # WDI: a fiscal year ending on or after 30 June is labelled by its ending year
        last = dt.date(label, m, d)
        first = dt.date(label - 1, m, d) + dt.timedelta(days=1)
    else:  # labelled by its starting year
        first = dt.date(label, m, d) + dt.timedelta(days=1)
        last = dt.date(label + 1, m, d)
    total = (last - first).days + 1
    w = {}
    for y in range(first.year, last.year + 1):
        a, b = max(first, dt.date(y, 1, 1)), min(last, dt.date(y, 12, 31))
        w[y] = ((b - a).days + 1) / total
    return w


def aligned(p: pd.DataFrame, fy: dict) -> pd.DataFrame:
    """Replace the four band logs of fiscal-year reporters by logs of day-weighted calendar means."""
    lv = (pd.read_csv(DATA / "country_ntl_annual.csv")[["gaul_code", "year", "avg_mean", "medmask_mean"]]
          .merge(pd.read_csv(DATA / "country_ntl_bands.csv")[["gaul_code", "year", "avgmask_mean", "median_mean"]],
                 on=["gaul_code", "year"]))
    lv = lv[lv.year >= VIIRS_START].set_index(["gaul_code", "year"])  # panel years only
    q = p.copy()
    rows = q.iso3.isin(fy)
    for i in q.index[rows]:
        w = weights(fy[q.at[i, "iso3"]], int(q.at[i, "year"]))
        keys = [(q.at[i, "gaul_code"], y) for y in w]
        for band, col in COLS.items():
            if all(k in lv.index for k in keys):
                v = sum(w[y] * lv.at[(q.at[i, "gaul_code"], y), col] for y in w)
                q.at[i, f"log_{band}"] = np.log(v) if v > 0 else np.nan
            else:
                q.at[i, f"log_{band}"] = np.nan
    return q


def summary(p: pd.DataFrame) -> dict:
    out = {}
    for band in ("median_masked", "average"):
        lad = ladder(p, band)
        out[band] = {"country_year_effects": [r6(v) for v in lad["fe"]], "annual_changes": [r6(v) for v in lad["annual"]],
                     "long_difference": [r6(v) for v in lad["ld"]], "economies": lad["countries"], "obs": lad["obs"]}
    h = heldout(p)
    out["heldout_one_year"] = {k: r6(v) for k, v in h[1].items()}
    return out


def main() -> None:
    fy = fiscal_reporters()
    p = panel(False)
    in_panel = sorted(set(p.iso3) & set(fy))
    al = aligned(p[p.iso3 != "MMR"], {k: v for k, v in fy.items() if k != "MMR"})
    out = {"fiscal_year_reporters_wdi": {k: f"{v[0]:02d}-{v[1]:02d}" for k, v in sorted(fy.items())},
           "panel_start": VIIRS_START,
           "fiscal_year_reporters_in_panel": in_panel,
           "aligned_left_out": ["MMR"],
           "example_weights": {"IND_2020": weights(fy["IND"], 2020), "PAK_2020": weights(fy["PAK"], 2020)},
           "calendar": summary(p),
           "excluded": summary(p[~p.iso3.isin(fy)]),
           "aligned": summary(al.dropna(subset=[f"log_{b}" for b in COLS], how="all")),
           "aligned_obs_dropped": int(al.loc[al.iso3.isin(fy), [f"log_{b}" for b in COLS]].isna().all(axis=1).sum())}
    with (BASE / "results_p0_fiscal_year.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    for k in ("calendar", "excluded", "aligned"):
        s = out[k]
        print(k, {b: (s[b]["country_year_effects"][0], s[b]["annual_changes"][0], s[b]["long_difference"][0], s[b]["economies"])
                  for b in ("median_masked", "average")}, {m: round(v, 4) for m, v in s["heldout_one_year"].items()})
    print("fiscal-year reporters in panel:", len(in_panel), in_panel)


if __name__ == "__main__":
    main()
