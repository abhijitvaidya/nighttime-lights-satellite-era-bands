"""Country-level annual DMSP-OLS extraction (1992-2013) for the global HSW
replication. ROBUST/UNATTENDED sharded design, identical code path to
the VIIRS extractor (single-threaded per shard + SIGALRM watchdog; GEE's
interactive quota silently hangs concurrent calls).

NOAA/DMSP-OLS/NIGHTTIME_LIGHTS 'stable_lights' (DN 0-63, HSW's product);
satellites flying the same calendar year are averaged (HSW's convention).

Run one process per shard:
    python extract_country_dmsp.py 0 3
    python extract_country_dmsp.py 1 3
    python extract_country_dmsp.py 2 3
Merge: python extract_country_dmsp.py --merge
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import os, sys, glob, signal, socket, time
import ee
from gaul_tiles import reduce_tiles, tile_geometries  # all GAUL tiles of a country
import pandas as pd

BASE = f"{_ROOT}/global_hsw"
CHK = f'{BASE}/data/checkpoints_dmsp'
OUT = f'{BASE}/data/country_dmsp_annual.csv'
YEARS = list(range(1992, 2014))
CALL_TIMEOUT = 75

class Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise Timeout()

def merge():
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f'{CHK}/shard*.csv'))]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(['gaul_code', 'year'])
    out.to_csv(OUT, index=False)
    err = int(out['stable_mean'].isna().sum())
    print(f'MERGED: {len(out)} rows ({err} null) -> {OUT}', flush=True)
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

    DMSP = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS')
    imgs = {y: DMSP.filterDate(f'{y}-01-01', f'{y}-12-31').select('stable_lights').mean()
            for y in YEARS}
    reducer = (ee.Reducer.mean().setOutputs(['mean'])
               .combine(ee.Reducer.count().setOutputs(['count']), '', True))

    done, rows = set(), []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev[prev['stable_mean'].notna()]
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
                # progressive watchdog: giants (Canada/China/Russia x multi-
                # satellite mean) legitimately need >75s; don't thrash retries
                signal.alarm(CALL_TIMEOUT * (attempt + 1))
                p = reduce_tiles(imgs[y], geoms, ['stable_lights'], scale=2000,
                                 crs='EPSG:4326', bestEffort=True,
                                 maxPixels=2_000_000, tileScale=4)
                signal.alarm(0)
                row = {'country': code2name[c], 'gaul_code': c, 'year': y,
                       'stable_mean': p.get('stable_lights_mean'),
                       'px_count': p.get('stable_lights_count')}
                break
            except Exception:
                signal.alarm(0)
                if attempt == 2:
                    row = {'country': code2name[c], 'gaul_code': c, 'year': y,
                           'stable_mean': None, 'px_count': None}
                else:
                    time.sleep(5)
        rows.append(row)
        n += 1
        if n % 25 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'shard {shard}: {n}/{len(tasks)}  [{time.time()-t0:.0f}s]', flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    nullc = sum(1 for r in rows if r['stable_mean'] is None)
    print(f'shard {shard} DONE: {len(rows)} rows ({nullc} null)  [{time.time()-t0:.0f}s]', flush=True)
