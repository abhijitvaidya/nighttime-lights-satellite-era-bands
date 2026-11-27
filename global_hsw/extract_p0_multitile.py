"""Country rows for the countries that GAUL-2015 stores as several features, from every tile.

FAO GAUL-2015 level 0 stores Canada (10 features), the United States (5), Australia (3) and the
West Bank (2) as several map tiles. This script reduces each tile with the settings and band
groupings of the country extractors and combines the tiles with `gaul_tiles.combine` (tile means
weighted by tile area in EPSG:4326), then replaces those countries' rows in:

  data/country_ntl_annual.csv   (average, median_masked; 2013-2025)
  data/country_ntl_bands.csv    (average_masked, median; 2013-2025)
  data/country_dmsp_annual.csv  (DMSP stable lights, mean of satellites; 1992-2013)
  data/country_ntl_monthly.csv  (VCMSLCFG avg_rad and cf_cvg; 2014-01 to 2025-12)

The country extractors use the same helper, so a full re-extraction gives the same rows; this
script exists because re-extracting four countries is much faster than re-extracting 275.
Per-tile results are cached in data/multitile_cache.jsonl (not bundled), so the combination can
be recomputed without Earth Engine. Requires EE_PROJECT. Run: python extract_p0_multitile.py
(faster: several `--shard i/n` processes at once, then one run without --shard to write the CSVs).
"""
from __future__ import annotations

import collections
import json
import os
import re
import signal
import socket
import sys
import time
import glob
from pathlib import Path

import ee
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))
from gaul_tiles import combine, reduce_tile  # noqa: E402

ee.Initialize(project=os.environ["EE_PROJECT"])
socket.setdefaulttimeout(300)
ee.data.setDeadline(300_000)  # Earth Engine request deadline: a hung call raises and is retried
countries = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.neq("ADM0_NAME", "Antarctica"))
meta = countries.reduceColumns(ee.Reducer.toList(4), ["system:index", "ADM0_CODE", "ADM0_NAME", "Shape_Area"]).get("list").getInfo()
names = {code: name for _, code, name, _ in meta}
MULTI = sorted(c for c, n in collections.Counter(code for _, code, _, _ in meta).items() if n > 1)
TILES = {c: [idx for idx, code, _, _ in meta if code == c] for c in MULTI}
AREA = {c: sum(a or 0 for _, code, _, a in meta if code == c) for c in MULTI}  # whole-country Shape_Area
print("multi-feature countries:", {names[c]: len(TILES[c]) for c in MULTI}, flush=True)
KW = dict(scale=2000, crs="EPSG:4326", bestEffort=True, maxPixels=2_000_000, tileScale=4)
PAIRS = {"ann": ["average", "median_masked"], "bnd": ["average_masked", "median"]}  # as in the two annual extractors

V21, V22 = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21"), ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V22")
DMSP = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS")
MONTHLY = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate("2014-01-01", "2025-12-31")
STACK = MONTHLY.map(lambda i: i.select(["avg_rad", "cf_cvg"]).rename(
    [ee.String("r_").cat(ee.Date(i.get("system:time_start")).format("YYYYMM")),
     ee.String("c_").cat(ee.Date(i.get("system:time_start")).format("YYYYMM"))])).toBands()
MBANDS = STACK.bandNames().getInfo()


def geom(idx, simplify_m):
    return ee.Feature(countries.filter(ee.Filter.eq("system:index", idx)).first()).simplify(simplify_m).geometry()


def monthly_scale(c):  # same tiers as extract_country_monthly.py, on the whole-country area
    a = AREA[c]
    return 1500 if a < 30 else (4000 if a < 300 else 10000)


def task(kind, c, idx, y=None):
    if kind in PAIRS:
        img = (V21 if y <= 2021 else V22).filterDate(f"{y}-01-01", f"{y}-12-31").first()
        return reduce_tile(img, geom(idx, 2000), PAIRS[kind], **KW)
    if kind == "dms":
        img = DMSP.filterDate(f"{y}-01-01", f"{y}-12-31").select("stable_lights").mean()
        return reduce_tile(img, geom(idx, 2000), ["stable_lights"], **KW)
    sc = monthly_scale(c)
    return reduce_tile(STACK, geom(idx, min(sc * 2, 15000)), MBANDS, scale=sc, crs="EPSG:4326", tileScale=4, maxPixels=1e13)


def _timeout(signum, frame):
    raise TimeoutError("task exceeded 600 s")


def run(key, args):
    """Run one task in a worker process with a hard time limit; Earth Engine errors are retried."""
    signal.signal(signal.SIGALRM, _timeout)
    for attempt in range(4):
        try:
            signal.alarm(600)
            out = task(*args)
            signal.alarm(0)
            return key, out
        except Exception as e:
            signal.alarm(0)
            print(f"retry {attempt + 1} {key}: {str(e)[:120]}", flush=True)
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def main() -> None:
    # One task per tile (and year): fewest tiles first, monthly stacks last.
    BY_SIZE = sorted(MULTI, key=lambda c: len(TILES[c]))
    tasks = {}
    for kind, years in (("ann", range(2013, 2026)), ("bnd", range(2013, 2026)), ("dms", range(1992, 2014))):
        for c in BY_SIZE:
            for idx in TILES[c]:
                for y in years:
                    tasks[f"{kind}|{c}|{idx}|{y}"] = (kind, c, idx, y)
    for c in BY_SIZE:
        for idx in TILES[c]:
            tasks[f"mon|{c}|{idx}"] = ("mon", c, idx)
    # Cache: data/multitile_cache.jsonl plus one file per shard. `--shard i/n` runs every n-th
    # remaining task from i, sequentially, in this process; several shards can run at once as
    # separate processes. Without --shard, all remaining tasks run here and the CSVs are written.
    shard = next((a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
                  for a in sys.argv[1:] if a.startswith("--shard")), None)
    CACHE = DATA / ("multitile_cache.jsonl" if shard is None else f"multitile_cache.shard{shard.replace('/', 'of')}.jsonl")
    done = {}
    for f in sorted(glob.glob(str(DATA / "multitile_cache*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            rec = json.loads(line)
            if rec["key"] in tasks:
                done[rec["key"]] = rec["result"]
    todo = {k: t for k, t in tasks.items() if k not in done}
    if shard is not None:
        i, n = map(int, shard.split("/"))
        todo = {k: t for j, (k, t) in enumerate(todo.items()) if j % n == i}
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    for n_done, (k, t) in enumerate(todo.items(), 1):
        done[k] = run(k, t)[1]
        with CACHE.open("a") as fh:
            fh.write(json.dumps({"key": k, "result": done[k]}) + "\n")
        if n_done % 25 == 0 or n_done == len(todo):
            print(f"{n_done}/{len(todo)} tasks of this run", flush=True)
    if shard is not None:
        return

    def country(kind, c, y=None):
        keys = [f"{kind}|{c}|{idx}|{y}" if y is not None else f"{kind}|{c}|{idx}" for idx in TILES[c]]
        bands = PAIRS.get(kind, ["stable_lights"] if kind == "dms" else MBANDS)
        return combine([done[k] for k in keys], bands)


    ann, bnd, dms, mon = [], [], [], []
    for c in BY_SIZE:
        for y in range(2013, 2026):
            a, b = country("ann", c, y), country("bnd", c, y)
            prod = "V21" if y <= 2021 else "V22"
            ann.append({"country": names[c], "gaul_code": c, "year": y, "product": prod,
                        "avg_mean": a["average_mean"], "avg_count": a["average_count"],
                        "medmask_mean": a["median_masked_mean"], "medmask_count": a["median_masked_count"]})
            bnd.append({"country": names[c], "gaul_code": c, "year": y,
                        "avgmask_mean": b["average_masked_mean"], "avgmask_count": b["average_masked_count"],
                        "median_mean": b["median_mean"], "median_count": b["median_count"]})
        for y in range(1992, 2014):
            d = country("dms", c, y)
            dms.append({"country": names[c], "gaul_code": c, "year": y,
                        "stable_mean": d["stable_lights_mean"], "px_count": d["stable_lights_count"]})
        m = country("mon", c)
        series = {}
        for b in MBANDS:
            g = re.search(r"([rc])_(\d{4})(\d{2})$", b)
            series.setdefault((int(g.group(2)), int(g.group(3))), {})[g.group(1)] = m[f"{b}_mean"]
        mon += [{"country": names[c], "gaul_code": c, "year": y, "month": mo,
                 "avg_rad_mean": v.get("r"), "cf_cvg_mean": v.get("c")} for (y, mo), v in sorted(series.items())]


    def replace(path: Path, new: list, keys: list) -> None:
        old = pd.read_csv(path)
        # other countries keep their original order; the re-extracted rows replace the old ones at the end
        new = pd.DataFrame(new)[old.columns].sort_values(keys)
        out = pd.concat([old[~old.gaul_code.isin(MULTI)], new], ignore_index=True)
        assert not out.duplicated(keys).any(), path.name
        out.to_csv(path, index=False)
        print(f"{path.name}: {len(old)} -> {len(out)} rows", flush=True)


    replace(DATA / "country_ntl_annual.csv", ann, ["gaul_code", "year"])
    replace(DATA / "country_ntl_bands.csv", bnd, ["gaul_code", "year"])
    replace(DATA / "country_dmsp_annual.csv", dms, ["gaul_code", "year"])
    replace(DATA / "country_ntl_monthly.csv", mon, ["gaul_code", "year", "month"])


if __name__ == "__main__":
    main()
