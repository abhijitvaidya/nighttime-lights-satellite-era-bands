"""Country-level annual VIIRS extraction for the global HSW replication.

ROBUST/UNATTENDED design. Concurrency wedges GEE's interactive-compute quota
(requests silently hang past ~100 calls; socket timeout doesn't fire), so this
runs SINGLE-THREADED and is sharded across a few independent processes instead.
A SIGALRM watchdog hard-aborts any getInfo() that exceeds CALL_TIMEOUT, so no
call can hang the run -- it retries, then records ERR and moves on.

Run one process per shard, e.g.:
    python extract_country_ntl.py 0 3   # shard 0 of 3
    python extract_country_ntl.py 1 3
    python extract_country_ntl.py 2 3
Each writes data/checkpoints_ntl/shard{i}.csv (checkpoint every 25). Merge with
    python extract_country_ntl.py --merge   (after all shards finish)

Annual VNL composites: ANNUAL_V21 (2013-2021) + ANNUAL_V22 (2022-2025), bands
'average' + 'median_masked' (stable-lights analogue). Country MEAN radiance at
2 km, geometry simplified to 2 km (raw GAUL polygons blow GEE's edge limit).
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import os, sys, glob, signal, socket, time
import ee
from gaul_tiles import reduce_tiles, tile_geometries  # all GAUL tiles of a country
import pandas as pd

BASE = f"{_ROOT}/global_hsw"
CHK = f'{BASE}/data/checkpoints_ntl'
OUT = f'{BASE}/data/country_ntl_annual.csv'
YEARS = list(range(2013, 2026))
CALL_TIMEOUT = 75   # seconds; a real call is 1-30s, so this only catches hangs

class Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise Timeout()

def merge():
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f'{CHK}/shard*.csv'))]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(['gaul_code', 'year'])
    out.to_csv(OUT, index=False)
    err = int((out['product'] == 'ERR').sum())
    print(f'MERGED: {len(out)} rows ({err} ERR) -> {OUT}', flush=True)
    return out

if __name__ == '__main__':
    if sys.argv[1:2] == ['--merge']:
        merge(); sys.exit(0)

    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(CHK, exist_ok=True)
    out_csv = f'{CHK}/shard{shard}.csv'
    socket.setdefaulttimeout(CALL_TIMEOUT)
    signal.signal(signal.SIGALRM, _alarm)
    ee.Initialize(project=os.environ['EE_PROJECT'])

    countries = (ee.FeatureCollection('FAO/GAUL/2015/level0')
                 .filter(ee.Filter.neq('ADM0_NAME', 'Antarctica')))
    clist = countries.aggregate_array('ADM0_CODE').getInfo()
    names = countries.aggregate_array('ADM0_NAME').getInfo()
    code2name = dict(zip(clist, names))
    mine = [c for i, c in enumerate(sorted(set(clist))) if i % nshard == shard]  # one entry per country
    tile_cache = {}

    V21 = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21')
    V22 = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V22')
    imgs = {y: (V21 if y <= 2021 else V22).filterDate(f'{y}-01-01', f'{y}-12-31')
               .first().select(['average', 'median_masked']) for y in YEARS}
    reducer = (ee.Reducer.mean().setOutputs(['mean'])
               .combine(ee.Reducer.count().setOutputs(['count']), '', True))

    done, rows = set(), []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev[prev['product'] != 'ERR']
        rows = prev.to_dict('records')
        done = set(zip(prev['gaul_code'], prev['year']))

    tasks = [(c, y) for c in mine for y in YEARS if (c, y) not in done]
    print(f'shard {shard}/{nshard}: {len(mine)} countries, {len(tasks)} tasks to do', flush=True)
    t0, n = time.time(), 0
    for c, y in tasks:
        if c not in tile_cache:
            tile_cache[c] = tile_geometries(countries, c, 2000)
        geoms = tile_cache[c]
        row = None
        for attempt in range(3):
            try:
                signal.alarm(CALL_TIMEOUT)
                p = reduce_tiles(imgs[y], geoms, ['average', 'median_masked'], scale=2000,
                                 crs='EPSG:4326', bestEffort=True,
                                 maxPixels=2_000_000, tileScale=4)
                signal.alarm(0)
                row = {'country': code2name[c], 'gaul_code': c, 'year': y,
                       'product': 'V21' if y <= 2021 else 'V22',
                       'avg_mean': p.get('average_mean'), 'avg_count': p.get('average_count'),
                       'medmask_mean': p.get('median_masked_mean'),
                       'medmask_count': p.get('median_masked_count')}
                break
            except Exception:
                signal.alarm(0)
                if attempt == 2:
                    row = {'country': code2name[c], 'gaul_code': c, 'year': y, 'product': 'ERR',
                           'avg_mean': None, 'avg_count': None, 'medmask_mean': None,
                           'medmask_count': None}
                else:
                    time.sleep(5)
        rows.append(row)
        n += 1
        if n % 25 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'shard {shard}: {n}/{len(tasks)}  [{time.time()-t0:.0f}s]', flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    err = sum(1 for r in rows if r['product'] == 'ERR')
    print(f'shard {shard} DONE: {len(rows)} rows ({err} ERR)  [{time.time()-t0:.0f}s]', flush=True)
