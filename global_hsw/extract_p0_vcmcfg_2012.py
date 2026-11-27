"""Unmasked 2012 and 2013 analogues from monthly VIIRS composites (Earth Engine).

The annual unmasked average and median bands for 2012 are not in Earth Engine. This script builds
annual analogues from the monthly VCMCFG composites (avg_rad), which begin in April 2012: the
mean and the median of the monthly images over April-December 2012 (2012ad), April 2012-March
2013 (2012am) and January-December 2013 (2013). Countries are reduced over all their GAUL-2015
tiles (gaul_tiles.py) with the settings of the annual extraction (2,000 m simplification and
scale, EPSG:4326, bestEffort). Only the within-source 2012-to-2013 change is used later.

Requires EE_PROJECT. Output: data/country_vcmcfg_2012_2013.csv
"""
from __future__ import annotations

import os
import signal
import socket
import sys
import time
from pathlib import Path

import ee
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from gaul_tiles import reduce_tiles, tile_geometries  # noqa: E402

OUT = BASE / "data" / "country_vcmcfg_2012_2013.csv"
PERIODS = {"2012ad": ("2012-04-01", "2013-01-01"), "2012am": ("2012-04-01", "2013-04-01"), "2013": ("2013-01-01", "2014-01-01")}


def main() -> None:
    ee.Initialize(project=os.environ["EE_PROJECT"])
    socket.setdefaulttimeout(300)
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
    fc = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.neq("ADM0_NAME", "Antarctica"))
    codes = fc.aggregate_array("ADM0_CODE").getInfo()
    names = dict(zip(codes, fc.aggregate_array("ADM0_NAME").getInfo()))
    ic = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").select("avg_rad")
    img, bands = None, []
    for k, (a, b) in PERIODS.items():
        sub = ic.filterDate(a, b)
        part = sub.mean().rename(f"mean_{k}").addBands(sub.median().rename(f"median_{k}"))
        img = part if img is None else img.addBands(part)
        bands += [f"mean_{k}", f"median_{k}"]
    months = {k: ic.filterDate(a, b).size().getInfo() for k, (a, b) in PERIODS.items()}
    assert months == {"2012ad": 9, "2012am": 12, "2013": 12}, months
    rows, done = [], set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows, done = prev.to_dict("records"), set(prev.gaul_code)
    for c in sorted(set(codes)):
        if c in done:
            continue
        geoms = tile_geometries(fc, c, 2000)
        rec = {"country": names[c], "gaul_code": c}
        for attempt in range(3):
            try:
                signal.alarm(120 * (attempt + 1))
                p = reduce_tiles(img, geoms, bands, scale=2000, crs="EPSG:4326", bestEffort=True,
                                 maxPixels=2_000_000, tileScale=4)
                signal.alarm(0)
                rec.update({b: p[f"{b}_mean"] for b in bands})
                break
            except Exception:
                signal.alarm(0)
                time.sleep(5)
        rows.append(rec)
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
    out = pd.DataFrame(rows).sort_values("gaul_code")
    out.to_csv(OUT, index=False)
    print("written", OUT, len(out), "missing", int(out[bands].isna().any(axis=1).sum()))


if __name__ == "__main__":
    main()
