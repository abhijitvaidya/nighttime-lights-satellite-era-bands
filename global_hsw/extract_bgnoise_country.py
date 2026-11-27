"""Pixel-level bgnoise BEFORE aggregation, server-side in GEE, many countries.

The country-mean monthly extraction cannot run PSTT's bgnoise_PSTT2021 step, because
that step masks dark pixels on the grid and must happen before spatial aggregation. This
script therefore
replicates bgnoise server-side (the xKDR rule, cf-weighted mean radiance of the
year's 12 monthly composites, threshold 0.4 nW/cm2/sr) and aggregates only over
lit pixels, so the pixel mask is applied before the country reduce.

For each country-year (VCMSLCFG monthly, 2014-2025) we compute, per pixel, the
cf-weighted annual mean radiance wm = sum(rad*cf)/sum(cf), then reduce over the
country to:
  raw    = mean(wm) over all pixels            [no mask; the unmasked 'average' analogue]
  lit    = mean(wm | wm>0.4)                    [bgnoise, per-year lit mask, lit-pixel mean]
  zero   = mean(wm with dark set to 0)          [bgnoise dark->0, EOG-masked-band style]
  stat   = mean(wm | wm_last>0.4)               [bgnoise with a single mask from the final year (2025), faithful
                                                 to bgnoise using the trailing 12 months]
  litfrac= fraction of valid pixels with wm>0.4

Output: data/country_bgnoise_annual.csv  (long: country x year x the 5 columns)
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import os, time, signal, socket
import ee
import pandas as pd

BASE = f"{_ROOT}/global_hsw"
OUT = f'{BASE}/data/country_bgnoise_annual.csv'
TH = 0.4
YEARS = list(range(2014, 2026))
MASK_YEAR = YEARS[-1]   # the static mask uses the window's final year

# ~20 countries chosen for variation across continents and lighting regimes:
# cloud-heavy tropical, bright urban core, sparse/dim, mid.
COUNTRIES = [
    # batch 1
    'India', 'Indonesia', 'Philippines', 'Viet Nam', 'Thailand', 'Bangladesh',
    'Nigeria', 'Kenya', 'Egypt', 'Democratic Republic of the Congo', 'Namibia',
    'Morocco', 'South Africa', 'Brazil', 'Mexico', 'Bolivia', 'Colombia',
    'Poland', 'Turkey', 'Mongolia', 'Kazakhstan',
    # batch 2: giants + more, across continents and income/LED regimes
    'China', 'Russian Federation', 'United States of America',
    'Pakistan', 'Iran  (Islamic Republic of)', 'Myanmar', 'Japan',
    'Republic of Korea', 'Malaysia', 'Saudi Arabia', 'Iraq', 'Afghanistan',
    'Ethiopia', 'United Republic of Tanzania', 'Angola', 'Algeria', 'Sudan',
    'Ghana', 'Mozambique', 'Zambia',
    'Argentina', 'Peru', 'Chile',
    'Spain', 'France', 'Germany', 'Italy', 'Ukraine',
]

class Timeout(Exception): pass
def _alarm(s, f): raise Timeout()
socket.setdefaulttimeout(300)
signal.signal(signal.SIGALRM, _alarm)
ee.Initialize(project=os.environ['EE_PROJECT'])

cs = ee.FeatureCollection('FAO/GAUL/2015/level0')
allnames = cs.aggregate_array('ADM0_NAME').getInfo()
miss = [c for c in COUNTRIES if c not in allnames]
if miss:
    print('WARNING unmatched GAUL names:', miss, flush=True)
areas = {n: a for n, a in zip(allnames, cs.aggregate_array('Shape_Area').getInfo())}

coll = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')

def wm_year(y):
    """Per-pixel cf-weighted annual mean radiance for year y."""
    c = coll.filterDate(f'{y}-01-01', f'{y}-12-31').select(['avg_rad', 'cf_cvg'])
    radcf = c.map(lambda im: im.select('avg_rad').multiply(im.select('cf_cvg'))).sum()
    cfsum = c.select('cf_cvg').sum()
    return radcf.divide(cfsum).rename('wm')          # masked where cfsum==0

def scale_for(area):
    if area < 10:   return 750
    if area < 50:   return 1500
    if area < 200:  return 3000
    return 6000

done, rows = set(), []
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    rows = prev.to_dict('records'); done = set(prev['country'].unique())
    print(f'resuming: {len(done)} countries done', flush=True)

# The United States polygon fails Earth Engine's limits for this per-pixel computation at
# every attempted scale (as in the original July 2026 run), so it is skipped rather than retried.
SKIP = {'United States of America'}

for name in COUNTRIES:
    if name in done or name in SKIP:
        continue
    geom = cs.filter(ee.Filter.eq('ADM0_NAME', name))
    sc = scale_for(areas.get(name, 100))
    mask_last = wm_year(MASK_YEAR).gt(TH)
    t0 = time.time()
    ok = 0
    for y in YEARS:
        wm = wm_year(y)
        lit = wm.gt(TH)
        img = (wm.rename('raw')
               .addBands(lit.rename('litfrac'))
               .addBands(wm.updateMask(lit).rename('lit'))
               .addBands(wm.where(lit.Not(), 0).rename('zero'))
               .addBands(wm.updateMask(mask_last).rename('stat')))
        props, used = None, sc
        for attempt in range(3):
            try:
                signal.alarm(180 * (attempt + 1))
                # a retry coarsens the scale; the scale actually used is recorded
                used = sc * (attempt + 1)
                r = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom.geometry(),
                                     scale=used, crs='EPSG:4326',
                                     maxPixels=1e10, tileScale=4).getInfo()
                signal.alarm(0); props = r; break
            except Exception as e:
                signal.alarm(0)
                if attempt == 2:
                    print(f'  {name} {y}: FAIL {type(e).__name__}', flush=True)
                else:
                    time.sleep(6)
        if props is None:
            continue
        rows.append({'country': name, 'year': y, 'scale_m': used,
                     'raw': props.get('raw'), 'lit': props.get('lit'),
                     'zero': props.get('zero'), 'stat': props.get('stat'),
                     'litfrac': props.get('litfrac')})
        ok += 1
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'{name}: {ok}/{len(YEARS)} years, scale {sc}m, {time.time()-t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f'\nwritten: {OUT} ({len(rows)} country-years)', flush=True)
