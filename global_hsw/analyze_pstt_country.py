"""PSTT-vs-masking test: can transparent statistical
cleaning of the MONTHLY series substitute for the VNL spatial mask?

Three country-level monthly variants (cleaning functions from nighttime_lights.py in the
repository root):
  raw         monthly avg_rad -> calendar-year mean
  hampel      cf_cvg==0 -> NaN; negatives -> NaN; Hampel(6, 3 MADs);
              interpolate -> annual mean        [the "fire/anomaly" steps]
  pstt_full   hampel + cf_cvg cloud-bias regression step  [full PSTT]

Each variant runs the levels 2wFE + growth design against WDI GDP, compared
with the four annual VNL bands. Hypothesis from the 2x2: the mask is SPATIAL
(removes persistently-noisy dark pixels) while PSTT is TEMPORAL, so PSTT
should recover at most part of the masked-band elasticity.

Output: results_pstt_country.json
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import glob, json, re, sys, unicodedata
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

sys.path.insert(0, f"{_ROOT}")
from nighttime_lights import outlier_hampel, na_interp_linear
from country_xwalk import gaul_key

BASE = f"{_ROOT}/global_hsw"

# ---------------- monthly data + variants ----------------
mon = pd.read_csv(f'{BASE}/data/country_ntl_monthly.csv')

def clean_country(g, bias_step=True):
    g = g.sort_values(['year', 'month']).copy()
    rad = g['avg_rad_mean'].astype(float).to_numpy()
    cf = g['cf_cvg_mean'].astype(float).to_numpy()
    rad = np.where((cf == 0) | np.isnan(cf), np.nan, rad)
    rad = np.where(rad < 0, np.nan, rad)
    rad = outlier_hampel(rad, window_size=6, n_sigmas=3)
    valid = ~np.isnan(rad) & ~np.isnan(cf)
    if bias_step and valid.sum() >= 12:
        slope, _ = np.polyfit(cf[valid], rad[valid], 1)
        rad = rad - slope * (cf - cf[valid].mean())
    return na_interp_linear(rad)

frames = []
for code, g in mon.groupby('gaul_code'):
    g = g.sort_values(['year', 'month']).copy()
    g['rad_hampel'] = clean_country(g, bias_step=False)
    g['rad_pstt'] = clean_country(g, bias_step=True)
    frames.append(g)
mon = pd.concat(frames, ignore_index=True)

ann = (mon.groupby(['country', 'gaul_code', 'year'])
       .agg(n_months=('avg_rad_mean', 'size'),
            raw=('avg_rad_mean', 'mean'),
            hampel=('rad_hampel', 'mean'),
            pstt=('rad_pstt', 'mean'))
       .reset_index())
ann = ann[ann['n_months'] >= 10]
print(f'annual from monthly: {len(ann)} country-years, {ann.gaul_code.nunique()} countries, '
      f'{ann.year.min()}-{ann.year.max()}')

# ---------------- WDI GDP (same matching as pilot) ----------------
def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z ]+', ' ', s)
    for t in ['islamic', 'republic', 'rep', 'democratic', 'dem', 'peoples', 'people',
              'of', 'the', 'state', 'states', 'plurinational', 'bolivarian',
              'federative', 'federal', 'arab', 'united']:
        s = re.sub(rf'\b{t}\b', '', s)
    return re.sub(r'\s+', '', s)

FIX = {'ivorycoast': 'cotedivoire', 'unitedstatesofamerica': 'unitedstates',
       'burma': 'myanmar', 'libyanjamahiriya': 'libya', 'swaziland': 'eswatini',
       'turkey': 'turkiye', 'czech': 'czechia', 'macedonia': 'northmacedonia',
       'formeryugoslavmacedonia': 'northmacedonia', 'capeverde': 'caboverde'}

ann['_n'] = gaul_key(ann['country'])

f = glob.glob(f'{BASE}/data/gdp_constant/API_*.csv')[0]
w = pd.read_csv(f, skiprows=4)
yrs = [c for c in w.columns if c.isdigit()]
gdp = w.melt(id_vars=['Country Name', 'Country Code'], value_vars=yrs,
             var_name='year', value_name='gdp')
gdp['year'] = gdp['year'].astype(int)
meta = pd.read_csv(glob.glob(f'{BASE}/data/gdp_constant/Metadata_Country_*.csv')[0])
gdp = gdp[gdp['Country Code'].isin(set(meta[meta['Region'].notna()]['Country Code']))]
gdp['_n'] = gdp['Country Code']

d = ann.merge(gdp[['_n', 'year', 'gdp']], on=['_n', 'year'], how='inner')
d = d[d['gdp'] > 0].copy()
d['log_y'] = np.log(d['gdp'])
ny = d.groupby('country')['year'].nunique()
d = d[d['country'].isin(ny[ny >= 8].index)]
print(f'panel: {len(d)} country-years, {d.country.nunique()} countries')

def two_specs(dd, xcol):
    s = dd[dd[xcol] > 0].copy()
    s['log_l'] = np.log(s[xcol])
    P = s.set_index(['country', 'year']).sort_index()
    m = PanelOLS(P['log_y'], P[['log_l']], entity_effects=True, time_effects=True
                 ).fit(cov_type='clustered', cluster_entity=True)
    g = s.sort_values(['country', 'year']).copy()
    g['_dy'] = g.groupby('country')['log_y'].diff()
    g['_dx'] = g.groupby('country')['log_l'].diff()
    g['_gap'] = g.groupby('country')['year'].diff()
    g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
    G = g.set_index(['country', 'year']).sort_index()
    m2 = PanelOLS(G['_dy'], G[['_dx']], time_effects=True
                  ).fit(cov_type='clustered', cluster_entity=True)
    return {'levels_2wfe': {'psi': float(m.params['log_l']), 'se': float(m.std_errors['log_l']),
                            'n': int(m.nobs)},
            'growth_yearfe': {'psi': float(m2.params['_dx']), 'se': float(m2.std_errors['_dx']),
                              'n': int(m2.nobs)},
            'dlogl_sd': float(g['_dx'].std())}

results = {}
print(f"\n{'variant':<14} {'levels 2wFE':>18} {'growth yrFE':>18} {'sd(dlogL)':>10}")
for col in ['raw', 'hampel', 'pstt']:
    r = two_specs(d, col)
    results[col] = r
    print(f"{col:<14} {r['levels_2wfe']['psi']:+.3f} ({r['levels_2wfe']['se']:.3f})"
          f"   {r['growth_yearfe']['psi']:+.3f} ({r['growth_yearfe']['se']:.3f})"
          f"   {r['dlogl_sd']:>9.3f}")

# reference rows from the annual-band analysis
try:
    B = json.load(open(f'{BASE}/results_bands_bridge.json'))['bands']
    for b in ['average', 'median_masked']:
        print(f"{'[' + b + ']':<14} {B[b]['levels_2wfe']['psi']:+.3f} ({B[b]['levels_2wfe']['se']:.3f})"
              f"   {B[b]['growth_yearfe']['psi']:+.3f} ({B[b]['growth_yearfe']['se']:.3f})")
except FileNotFoundError:
    pass

with open(f'{BASE}/results_pstt_country.json', 'w') as fo:
    json.dump(results, fo, indent=2)
print('\nwritten: results_pstt_country.json')
