"""Complementary VNL bands for the 2x2 band decomposition:

            unmasked        masked
  mean      average*        average_masked   <- this script
  median    median          median_masked*   <- this script
  (* already extracted by extract_country_ntl.py)

Decomposes the psi gap between 'average' and 'median_masked' into the
masking effect (background/dim-pixel zeros) vs the statistic effect
(median kills ephemeral lights: fires, flares, boats).

Same robust sharded design as the other extractors (single-threaded shards +
SIGALRM watchdog). Run:
    python extract_country_ntl_bands.py {0,1,2} 3
Merge: python extract_country_ntl_bands.py --merge
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import os, sys, glob, signal, socket, time
import ee
from gaul_tiles import reduce_tiles, tile_geometries  # all GAUL tiles of a country
import pandas as pd

BASE = f"{_ROOT}/global_hsw"
CHK = f'{BASE}/data/checkpoints_bands'
OUT = f'{BASE}/data/country_ntl_bands.csv'
YEARS = list(range(2013, 2026))
CALL_TIMEOUT = 75

class Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise Timeout()

def merge():
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f'{CHK}/shard*.csv'))]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(['gaul_code', 'year'])
    out.to_csv(OUT, index=False)
    err = int(out['avgmask_mean'].isna().sum())
    print(f'MERGED: {len(out)} rows ({err} null) -> {OUT}', flush=True)

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
               .first().select(['average_masked', 'median']) for y in YEARS}
    reducer = (ee.Reducer.mean().setOutputs(['mean'])
               .combine(ee.Reducer.count().setOutputs(['count']), '', True))

    done, rows = set(), []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev[prev['avgmask_mean'].notna()]
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
                signal.alarm(CALL_TIMEOUT * (attempt + 1))   # progressive: giants need >75s
                p = reduce_tiles(imgs[y], geoms, ['average_masked', 'median'], scale=2000,
                                 crs='EPSG:4326', bestEffort=True,
                                 maxPixels=2_000_000, tileScale=4)
                signal.alarm(0)
                row = {'country': code2name[c], 'gaul_code': c, 'year': y,
                       'avgmask_mean': p.get('average_masked_mean'),
                       'avgmask_count': p.get('average_masked_count'),
                       'median_mean': p.get('median_mean'),
                       'median_count': p.get('median_count')}
                break
            except Exception:
                signal.alarm(0)
                if attempt == 2:
                    row = {'country': code2name[c], 'gaul_code': c, 'year': y,
                           'avgmask_mean': None, 'avgmask_count': None,
                           'median_mean': None, 'median_count': None}
                else:
                    time.sleep(5)
        rows.append(row)
        n += 1
        if n % 25 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'shard {shard}: {n}/{len(tasks)}  [{time.time()-t0:.0f}s]', flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    nullc = sum(1 for r in rows if r['avgmask_mean'] is None)
    print(f'shard {shard} DONE: {len(rows)} rows ({nullc} null)  [{time.time()-t0:.0f}s]', flush=True)
