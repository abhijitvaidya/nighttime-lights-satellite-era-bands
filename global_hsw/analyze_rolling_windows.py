"""Rolling-window Henderson elasticities, 1992-2025: is the VIIRS-era decline
a sensor change or a regime change?

For every 12-year window in the DMSP era (1992-2003 ... 2002-2013) and the
VIIRS window (2014-2025, the main panel), estimate on WDI constant-price GDP:
  (a) country + year FE levels elasticity
  (b) HSW growth design (annual dlog, year FE, consecutive pairs)
  (c) long difference (first -> last year of window)
Identical code across eras/sensors; only the lights series differs.
Output: results_rolling.json + console table.
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import json, glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

BASE = f"{_ROOT}/global_hsw"
WIN = 12  # paired years per window

# ---------------- WDI GDP (long) ----------------
f = glob.glob(f'{BASE}/data/gdp_constant/API_*.csv')[0]
w = pd.read_csv(f, skiprows=4)
yrs = [c for c in w.columns if c.isdigit()]
gdp = w.melt(id_vars=['Country Name', 'Country Code'], value_vars=yrs,
             var_name='year', value_name='gdp')
gdp['year'] = gdp['year'].astype(int)
meta = pd.read_csv(glob.glob(f'{BASE}/data/gdp_constant/Metadata_Country_*.csv')[0])
gdp = gdp[gdp['Country Code'].isin(set(meta[meta['Region'].notna()]['Country Code']))]
gdp = gdp.rename(columns={'Country Name': 'wb_name'})

# ---------------- name match (reuse pilot logic) ----------------
import re, unicodedata
from country_xwalk import gaul_key
from p0_config import VIIRS_START
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

def attach_names(df):
    df['_n'] = gaul_key(df['country'])
    return df

gdp['_n'] = gdp['Country Code']

# ---------------- lights panels ----------------
dmsp = attach_names(pd.read_csv(f'{BASE}/data/country_dmsp_annual.csv'))
dmsp['light'] = dmsp['stable_mean']
viirs = attach_names(pd.read_csv(f'{BASE}/data/country_ntl_annual.csv'))
viirs = viirs[viirs['year'] >= VIIRS_START]  # main VIIRS panel (p0_config.py)
# median_masked is the stable-lights analogue -> the band-comparable series;
# the noisy 'average' band is reported as a separate row for contrast.
viirs_mm = viirs.copy(); viirs_mm['light'] = viirs_mm['medmask_mean']
viirs_av = viirs.copy(); viirs_av['light'] = viirs_av['avg_mean']

def build(lights):
    d = lights.merge(gdp[['_n', 'year', 'gdp']], on=['_n', 'year'], how='inner')
    d = d[(d['light'] > 0) & (d['gdp'] > 0)].copy()
    d['log_y'] = np.log(d['gdp'])
    d['log_l'] = np.log(d['light'])
    return d

panels = {'DMSP': build(dmsp), 'VIIRSmm': build(viirs_mm), 'VIIRSav': build(viirs_av)}

def window_est(d, y0, y1):
    s = d[(d['year'] >= y0) & (d['year'] <= y1)].copy()
    ny = s.groupby('country')['year'].nunique()
    s = s[s['country'].isin(ny[ny >= max(8, (y1 - y0 + 1) - 3)].index)]
    if s['country'].nunique() < 50:
        return None
    P = s.set_index(['country', 'year']).sort_index()
    out = {'n_countries': int(s['country'].nunique()), 'obs': len(s)}
    m = PanelOLS(P['log_y'], P[['log_l']], entity_effects=True, time_effects=True
                 ).fit(cov_type='clustered', cluster_entity=True)
    out['level_2wfe'] = {'psi': float(m.params['log_l']), 'se': float(m.std_errors['log_l'])}
    g = s.sort_values(['country', 'year']).copy()
    g['_dy'] = g.groupby('country')['log_y'].diff()
    g['_dx'] = g.groupby('country')['log_l'].diff()
    g['_gap'] = g.groupby('country')['year'].diff()
    g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
    G = g.set_index(['country', 'year']).sort_index()
    m2 = PanelOLS(G['_dy'], G[['_dx']], time_effects=True
                  ).fit(cov_type='clustered', cluster_entity=True)
    out['growth_yearfe'] = {'psi': float(m2.params['_dx']), 'se': float(m2.std_errors['_dx'])}
    a = s[s['year'] == y0].set_index('country')
    b = s[s['year'] == y1].set_index('country')
    common = a.index.intersection(b.index)
    if len(common) >= 50:
        dy = b.loc[common, 'log_y'] - a.loc[common, 'log_y']
        dx = b.loc[common, 'log_l'] - a.loc[common, 'log_l']
        ml = sm.OLS(dy, sm.add_constant(pd.DataFrame({'d': dx}))).fit(cov_type='HC1')
        out['long_diff'] = {'psi': float(ml.params['d']), 'se': float(ml.bse['d']),
                            'n': int(ml.nobs)}
    return out

results = {}
print(f"{'window':>12} {'sensor':>6} {'2wFE':>16} {'growth+yrFE':>16} {'long-diff':>16} {'nc':>4}")
for sensor, dd in panels.items():
    lo, hi = int(dd['year'].min()), int(dd['year'].max())
    for y0 in range(lo, hi - WIN + 2):
        y1 = y0 + WIN - 1
        if y1 > hi:
            continue
        r = window_est(dd, y0, y1)
        if r is None:
            continue
        key = f'{sensor}_{y0}_{y1}'
        results[key] = r
        f1 = f"{r['level_2wfe']['psi']:+.3f}({r['level_2wfe']['se']:.3f})"
        f2 = f"{r['growth_yearfe']['psi']:+.3f}({r['growth_yearfe']['se']:.3f})"
        f3 = (f"{r['long_diff']['psi']:+.3f}({r['long_diff']['se']:.3f})"
              if 'long_diff' in r else '--')
        print(f'{y0}-{y1:>6} {sensor:>6} {f1:>16} {f2:>16} {f3:>16} {r["n_countries"]:>4}')

with open(f'{BASE}/results_rolling.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nwritten: results_rolling.json')
