"""Does the bgnoise-vs-masked-band gap narrow as the pixel mask runs at finer scale?

The 48-country test ran the server-side bgnoise mask at 0.75-6 km (coarse for big
countries). If the residual gap to EOG's native ~500 m masked band is a resolution
effect, then applying bgnoise at progressively finer scale should move the series
closer to the masked band. This re-extracts a handful of SMALL countries (cheap at
500 m) at scales 2000 / 1000 / 500 m and records raw + bgnoise-static per year.

The reduceRegion scale sets the pixel size at which the wm image and the 0.4 mask
are evaluated, so sweeping scale sweeps the effective mask resolution.

Output: data/country_bgnoise_scale.csv  (country x year x scale: raw, stat, litfrac)
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import os, time, signal, socket
import ee
import pandas as pd

BASE = f"{_ROOT}/global_hsw"
OUT = f'{BASE}/data/country_bgnoise_scale.csv'
TH = 0.4
YEARS = list(range(2014, 2026))
MASK_YEAR = YEARS[-1]   # the static mask uses the window's final year
SCALES = [2000, 1000, 500]
# small countries: cheap at 500 m, spread across income and cloud regimes
COUNTRIES = ['Netherlands', 'Belgium', 'Rwanda', 'Sri Lanka', 'Nepal', 'Uruguay',
             'Ecuador', 'Malawi']

class Timeout(Exception): pass
def _alarm(s, f): raise Timeout()
socket.setdefaulttimeout(300)
signal.signal(signal.SIGALRM, _alarm)
ee.Initialize(project=os.environ['EE_PROJECT'])

cs = ee.FeatureCollection('FAO/GAUL/2015/level0')
alln = set(cs.aggregate_array('ADM0_NAME').getInfo())
miss = [c for c in COUNTRIES if c not in alln]
if miss:
    print('WARNING unmatched:', miss, flush=True)
coll = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')

def wm_year(y):
    c = coll.filterDate(f'{y}-01-01', f'{y}-12-31').select(['avg_rad', 'cf_cvg'])
    radcf = c.map(lambda im: im.select('avg_rad').multiply(im.select('cf_cvg'))).sum()
    cfsum = c.select('cf_cvg').sum()
    return radcf.divide(cfsum).rename('wm')

done, rows = set(), []
if os.path.exists(OUT):
    prev = pd.read_csv(OUT); rows = prev.to_dict('records')
    done = set(zip(prev['country'], prev['scale']))

for name in COUNTRIES:
    geom = cs.filter(ee.Filter.eq('ADM0_NAME', name)).geometry()
    for sc in SCALES:
        if (name, sc) in done:
            continue
        t0, ok = time.time(), 0
        mask_last = wm_year(MASK_YEAR).gt(TH)
        for y in YEARS:
            wm = wm_year(y)
            lit = wm.gt(TH)
            img = (wm.rename('raw')
                   .addBands(lit.rename('litfrac'))
                   .addBands(wm.updateMask(mask_last).rename('stat')))
            props = None
            for attempt in range(3):
                try:
                    signal.alarm(240 * (attempt + 1))
                    r = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom,
                                         scale=sc, crs='EPSG:4326', maxPixels=1e10,
                                         tileScale=4 * (attempt + 1)).getInfo()
                    signal.alarm(0); props = r; break
                except Exception as e:
                    signal.alarm(0)
                    if attempt == 2:
                        print(f'  {name} {y} @{sc}m: FAIL {type(e).__name__}', flush=True)
                    else:
                        time.sleep(6)
            if props is None:
                continue
            rows.append({'country': name, 'year': y, 'scale': sc,
                         'raw': props.get('raw'), 'stat': props.get('stat'),
                         'litfrac': props.get('litfrac')})
            ok += 1
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f'{name} @{sc}m: {ok}/{len(YEARS)} years, {time.time()-t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f'\nwritten: {OUT} ({len(rows)} rows)', flush=True)
