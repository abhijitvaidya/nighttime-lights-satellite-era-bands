"""Country means of DMSP stable lights for each satellite-year, 1992-2013 (Earth Engine).

The main DMSP series (`extract_country_dmsp.py`) averages the satellites that observe the same
year, following HSW. Gibson, Olivia and Boe-Gibson (2020) recommend satellite effects instead,
because DMSP sensors differ in gain and calibration. This script keeps each satellite-year
separately: the 35 images of NOAA/DMSP-OLS/NIGHTTIME_LIGHTS for 1992-2013 (F10 to F18) are
stacked as bands and reduced once per GAUL-2015 tile at a fixed 2 km scale in EPSG:4326
(no bestEffort, so every tile is evaluated at the same scale), with the 2,000 m geometry
simplification of the country extractors. Tiles are combined by area (`gaul_tiles.combine`).

Run: several `--shard i/n` processes, then one run without --shard to write the CSV.
Per-tile results are cached in data/dmsp_satellite_cache*.jsonl (not bundled).
Requires EE_PROJECT. Output: data/country_dmsp_satellite.csv
"""
from __future__ import annotations

import glob
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import ee
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))
from gaul_tiles import combine, reduce_tile  # noqa: E402

ee.Initialize(project=os.environ["EE_PROJECT"])
socket.setdefaulttimeout(600)
ee.data.setDeadline(600_000)
countries = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.neq("ADM0_NAME", "Antarctica"))
meta = countries.reduceColumns(ee.Reducer.toList(4), ["system:index", "ADM0_CODE", "ADM0_NAME", "Shape_Area"]).get("list").getInfo()
COL = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS").filterDate("1992-01-01", "2014-01-01")
IDS = COL.aggregate_array("system:index").getInfo()
STACK = ee.Image.cat([ee.Image(COL.filter(ee.Filter.eq("system:index", i)).first()).select("stable_lights").rename(i) for i in IDS])


def _timeout(signum, frame):
    raise TimeoutError("task exceeded 900 s")


def task(idx):
    g = ee.Feature(countries.filter(ee.Filter.eq("system:index", idx)).first()).simplify(2000).geometry()
    return reduce_tile(STACK, g, IDS, scale=2000, crs="EPSG:4326", maxPixels=1e13, tileScale=4)


def run(idx):
    signal.signal(signal.SIGALRM, _timeout)
    for attempt in range(4):
        try:
            signal.alarm(900)
            out = task(idx)
            signal.alarm(0)
            return out
        except Exception as e:
            signal.alarm(0)
            print(f"retry {attempt + 1} {idx}: {str(e)[:120]}", flush=True)
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))


def main() -> None:
    shard = next((a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
                  for a in sys.argv[1:] if a.startswith("--shard")), None)
    tiles = [m for m in sorted(meta, key=lambda r: r[3] or 0)]  # smallest first
    done = {}
    for f in sorted(glob.glob(str(DATA / "dmsp_satellite_cache*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            rec = json.loads(line)
            done[rec["key"]] = rec["result"]
    todo = [t for t in tiles if t[0] not in done]
    if shard is not None:
        i, n = map(int, shard.split("/"))
        todo = [t for j, t in enumerate(todo) if j % n == i]
    cache = DATA / ("dmsp_satellite_cache.jsonl" if shard is None else f"dmsp_satellite_cache.shard{shard.replace('/', 'of')}.jsonl")
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    for k, (idx, code, name, _) in enumerate(todo, 1):
        done[idx] = run(idx)
        with cache.open("a") as fh:
            fh.write(json.dumps({"key": idx, "result": done[idx]}) + "\n")
        if k % 25 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} tiles of this run", flush=True)
    if shard is not None:
        return
    rows = []
    by_code = {}
    for idx, code, name, _ in meta:
        by_code.setdefault((code, name), []).append(idx)
    for (code, name), idxs in sorted(by_code.items()):
        c = combine([done[i] for i in idxs], IDS)
        for i in IDS:
            rows.append({"country": name, "gaul_code": code, "satellite": i[:3], "year": int(i[3:]),
                         "stable_mean": c[f"{i}_mean"], "px_count": c[f"{i}_count"]})
    out = pd.DataFrame(rows).sort_values(["gaul_code", "year", "satellite"])
    out.to_csv(DATA / "country_dmsp_satellite.csv", index=False)
    print("written", len(out), "rows;", out.gaul_code.nunique(), "countries;", len(IDS), "satellite-years", flush=True)


if __name__ == "__main__":
    main()
