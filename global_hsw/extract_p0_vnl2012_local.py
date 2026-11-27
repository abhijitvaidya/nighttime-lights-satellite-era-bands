"""Country means of the EOG VNL V2.1 masked bands for 2012 and 2013, computed locally.

Earth Engine's annual VNL collection starts in 2013. EOG distributes two composites labelled 2012:
April-December 2012 (`201204-201212`) and April 2012-March 2013 (`201204-201303`), both with
average_masked and median_masked. This script reduces those rasters and the 2013 rasters to
country means on the same geography as the Earth Engine extraction: every GAUL-2015 level-0
feature of a country (see gaul_tiles.py), simplified at 2,000 m, with the mean taken over all
15-arc-second pixels whose centres fall inside the country (background zeros included).

The rasters require a free EOG account and are not bundled. Download, without renaming, into
NTL_EOG_DIR (default: ../../data/eog_2012 relative to this folder):
  VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.{average_masked,median_masked}.dat.tif.gz
  VNL_v21_npp_201204-201303_global_vcmcfg_c202205302300.{average_masked,median_masked}.dat.tif.gz
  VNL_v21_npp_2013_global_vcmcfg_c202205302300.{average_masked,median_masked}.dat.tif.gz
from https://eogdata.mines.edu/nighttime_light/annual/v21/2012/ and .../2013/.

Requires EE_PROJECT (for the GAUL geometries), rasterio and shapely.
Output: data/country_vnl2012_masked.csv (country x period x band: mean, pixel count).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box, shape

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
EOG = Path(os.environ.get("NTL_EOG_DIR", BASE.parent.parent / "data" / "eog_2012"))
WORK = EOG / "work"
OUT = DATA / "country_vnl2012_masked.csv"
PERIODS = {"201204-201212": "2012ad", "201204-201303": "2012am", "2013": "2013"}
STRIP = 1024


def fetch_tiles() -> list[dict]:
    """All GAUL level-0 features (tiles), simplified at 2,000 m, smallest country first; cached."""
    import ee
    WORK.mkdir(parents=True, exist_ok=True)
    cache = WORK / "gaul_tiles_simplify2000.jsonl"
    have = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            f = json.loads(line)
            have[f["properties"]["tile"]] = f
    ee.Initialize(project=os.environ["EE_PROJECT"])
    socket.setdefaulttimeout(900)
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
    fc = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.neq("ADM0_NAME", "Antarctica"))
    meta = fc.reduceColumns(ee.Reducer.toList(4), ["system:index", "ADM0_CODE", "ADM0_NAME", "Shape_Area"]).get("list").getInfo()
    queue = [m for m in sorted(meta, key=lambda r: r[3]) if m[0] not in have]
    slow = []
    for limit, items in ((45, queue), (900, None)):
        for idx, code, name, _ in (items if items is not None else slow):
            signal.alarm(limit)
            try:
                g = ee.Feature(fc.filter(ee.Filter.eq("system:index", idx)).first()).simplify(2000).geometry().getInfo()
                signal.alarm(0)
            except Exception:
                signal.alarm(0)
                if items is not None:
                    slow.append((idx, code, name, None))
                    continue
                raise
            f = {"type": "Feature", "properties": {"tile": idx, "gaul_code": code, "country": name}, "geometry": g}
            have[idx] = f
            with cache.open("a") as fh:
                fh.write(json.dumps(f) + "\n")
    assert len(have) == len(meta), "GAUL tiles missing"
    return list(have.values())


def main() -> None:
    tiles = fetch_tiles()
    codes = sorted({t["properties"]["gaul_code"] for t in tiles})
    index = {c: i + 1 for i, c in enumerate(codes)}
    names = {t["properties"]["gaul_code"]: t["properties"]["country"] for t in tiles}
    geoms = [(shape(t["geometry"]), index[t["properties"]["gaul_code"]]) for t in tiles]
    files = sorted(glob.glob(str(EOG / "VNL_v21_npp_*_global_vcmcfg_*.*_masked.dat.tif.gz")))
    assert len(files) == 6, f"expected 6 EOG masked rasters in {EOG}, found {len(files)}"
    with rasterio.open("/vsigzip/" + files[0]) as r:
        W, H, T = r.width, r.height, r.transform
    key = hashlib.sha256(json.dumps(sorted(t["properties"]["tile"] for t in tiles)).encode()).hexdigest()[:12]
    idp = WORK / f"country_ids_{W}x{H}_{key}.u16"
    if not idp.exists():
        ids = np.memmap(str(idp) + ".tmp", dtype=np.uint16, mode="w+", shape=(H, W))
        for r0 in range(0, H, STRIP):
            h = min(STRIP, H - r0)
            top = T.f + T.e * r0
            band = box(-180, top + T.e * h, 180, top)
            shp = [(g, v) for g, v in geoms if g.intersects(band)]
            ids[r0:r0 + h] = (rasterize(shp, out_shape=(h, W), transform=rasterio.windows.transform(Window(0, r0, W, h), T),
                                        fill=0, dtype="uint16") if shp else 0)
        ids.flush()
        del ids
        os.rename(str(idp) + ".tmp", idp)
    ids = np.memmap(idp, dtype=np.uint16, mode="r", shape=(H, W))
    rows = []
    for f in files:
        name = os.path.basename(f)
        period = PERIODS[name.split("_global")[0].replace("VNL_v21_npp_", "")]
        band = name.split(".")[1]
        s = np.zeros(len(codes) + 1)
        n = np.zeros(len(codes) + 1)
        t0 = time.time()
        with rasterio.open("/vsigzip/" + f) as r:
            assert (r.width, r.height) == (W, H) and r.transform == T
            for r0 in range(0, H, STRIP):
                h = min(STRIP, H - r0)
                v = r.read(1, window=Window(0, r0, W, h)).ravel()
                k = ids[r0:r0 + h].ravel()
                ok = np.isfinite(v)
                s += np.bincount(k[ok], weights=v[ok], minlength=len(codes) + 1)
                n += np.bincount(k[ok], minlength=len(codes) + 1)
        for c in codes:
            i = index[c]
            rows.append({"country": names[c], "gaul_code": c, "period": period, "band": band,
                         "mean": round(float(s[i] / n[i]), 8) if n[i] else None, "count": int(n[i])})
        print(period, band, f"{time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows).sort_values(["gaul_code", "period", "band"])
    out.to_csv(OUT, index=False)
    print("written", OUT, len(out))


if __name__ == "__main__":
    sys.exit(main())
