"""Country-level MONTHLY VIIRS (avg_rad + cf_cvg), 2014-01..2025-12, for the
PSTT-vs-masking test. SINGLE PROCESS, one call per GAUL tile (~25 min).

Key design: stack all 144 monthly images into ONE 288-band image (r_YYYYMM /
c_YYYYMM), then ONE reduceRegion per country returns the whole monthly series.
This collapses 291x144 = 42k per-(tile,month) calls into 291 per-tile
calls. Scale is tiered by country area (Shape_Area, sq.deg) so giant polygons
(Russia 17s @10km) stay fast while small bright countries keep a fine scale.
Per-country SIGALRM watchdog + checkpoint every 20 countries; resumable.

Output: data/country_ntl_monthly.csv  (long: country x month, avg_rad + cf_cvg)
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import os, re, signal, socket, time
import ee
from gaul_tiles import reduce_tiles, tile_geometries  # all GAUL tiles of a country
import pandas as pd

BASE = f"{_ROOT}/global_hsw"
OUT = f'{BASE}/data/country_ntl_monthly.csv'

class Timeout(Exception):
    pass

def _alarm(s, f):
    raise Timeout()

socket.setdefaulttimeout(300)
signal.signal(signal.SIGALRM, _alarm)
ee.Initialize(project=os.environ['EE_PROJECT'])

cs = ee.FeatureCollection('FAO/GAUL/2015/level0').filter(
    ee.Filter.neq('ADM0_NAME', 'Antarctica'))
meta = cs.aggregate_array('ADM0_CODE').getInfo()
names = cs.aggregate_array('ADM0_NAME').getInfo()
areas = cs.aggregate_array('Shape_Area').getInfo()
info = {}
for c, n, a in zip(meta, names, areas):  # area of the whole country: sum over its GAUL tiles
    info[c] = (n, info.get(c, (n, 0))[1] + (a or 0))

coll = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterDate(
    '2014-01-01', '2025-12-31')
def tag(img):
    d = ee.Date(img.get('system:time_start')).format('YYYYMM')
    return img.select(['avg_rad', 'cf_cvg']).rename(
        [ee.String('r_').cat(d), ee.String('c_').cat(d)])
stack = coll.map(tag).toBands()
print(f'stack bands: {stack.bandNames().size().getInfo()}; {len(info)} countries', flush=True)

def scale_for(area):
    return 1500 if area < 30 else (4000 if area < 300 else 10000)

done, rows = set(), []
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    rows = prev.to_dict('records')
    done = set(prev['gaul_code'].unique())
    print(f'resuming: {len(done)} countries done', flush=True)

todo = [c for c in dict.fromkeys(meta) if c not in done]  # one entry per country
BANDS = stack.bandNames().getInfo()
t0, n = time.time(), 0
for code in todo:
    name, area = info[code]
    sc = scale_for(area)
    geoms = tile_geometries(cs, code, min(sc * 2, 15000))
    props = None
    for attempt in range(3):
        try:
            signal.alarm(120 * (attempt + 1))
            r = reduce_tiles(stack, geoms, BANDS, scale=sc * (attempt + 1),
                             crs='EPSG:4326', tileScale=4, maxPixels=1e13)
            signal.alarm(0)
            props = {k[:-5]: v for k, v in r.items() if k.endswith('_mean')}
            break
        except Exception as e:
            signal.alarm(0)
            if attempt == 2:
                print(f'{name}: FAIL {type(e).__name__}', flush=True)
            else:
                time.sleep(8)
    if props is None:
        rows.append({'country': name, 'gaul_code': code, 'year': None, 'month': None,
                     'avg_rad_mean': None, 'cf_cvg_mean': None})
        continue
    series = {}
    for k, v in props.items():
        m = re.search(r'_([rc])_(\d{4})(\d{2})$', k)
        if not m:
            continue
        var, y, mo = m.group(1), int(m.group(2)), int(m.group(3))
        series.setdefault((y, mo), {})[var] = v
    for (y, mo), d in sorted(series.items()):
        rows.append({'country': name, 'gaul_code': code, 'year': y, 'month': mo,
                     'avg_rad_mean': d.get('r'), 'cf_cvg_mean': d.get('c')})
    n += 1
    if n % 20 == 0:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f'{n}/{len(todo)} countries  [{time.time()-t0:.0f}s]', flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f'DONE: {len(rows)} country-months, {pd.DataFrame(rows).gaul_code.nunique()} countries '
      f'[{time.time()-t0:.0f}s]', flush=True)
