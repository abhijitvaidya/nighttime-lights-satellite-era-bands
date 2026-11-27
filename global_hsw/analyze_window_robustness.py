"""Window-length robustness for the 'annual growth elasticity was never 0.3' claim.

The headline rolling-window analysis (analyze_rolling_windows.py) fixes the
estimation window at 12 years so DMSP windows match the VIIRS span. This script checks
that 12 is not special: the annual-growth design differences over a
1-year horizon regardless of window length, so the window only sets how many
annual pairs feed each regression. This script sweeps the window length L from 3
to 18 and reports, for every L and sensor, the range of the annual-growth
elasticity across all rolling positions, alongside the within-window long
difference (which DOES depend on L and should rise with the horizon).

Identical estimation code to analyze_rolling_windows.py; only WIN varies.
Output: results_window_robustness.json + console table.
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import json, glob, math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

BASE = f"{_ROOT}/global_hsw"
LMIN, LMAX = 3, 18

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

dmsp = attach_names(pd.read_csv(f'{BASE}/data/country_dmsp_annual.csv'))
dmsp['light'] = dmsp['stable_mean']
viirs = attach_names(pd.read_csv(f'{BASE}/data/country_ntl_annual.csv'))
viirs = viirs[viirs['year'] >= VIIRS_START]  # main VIIRS panel (p0_config.py)
viirs_mm = viirs.copy(); viirs_mm['light'] = viirs_mm['medmask_mean']

def build(lights):
    d = lights.merge(gdp[['_n', 'year', 'gdp']], on=['_n', 'year'], how='inner')
    d = d[(d['light'] > 0) & (d['gdp'] > 0)].copy()
    d['log_y'] = np.log(d['gdp']); d['log_l'] = np.log(d['light'])
    return d

panels = {'DMSP': build(dmsp), 'VIIRSmm': build(viirs_mm)}

def growth_and_ld(d, y0, y1):
    """Annual growth elasticity (year FE) and first->last long diff on window."""
    L = y1 - y0 + 1
    s = d[(d['year'] >= y0) & (d['year'] <= y1)].copy()
    need = max(3, math.ceil(0.75 * L))         # >=75% of window years, floor 3
    ny = s.groupby('country')['year'].nunique()
    s = s[s['country'].isin(ny[ny >= need].index)]
    if s['country'].nunique() < 50:
        return None
    out = {'n_countries': int(s['country'].nunique())}
    g = s.sort_values(['country', 'year']).copy()
    g['_dy'] = g.groupby('country')['log_y'].diff()
    g['_dx'] = g.groupby('country')['log_l'].diff()
    g['_gap'] = g.groupby('country')['year'].diff()
    g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
    if g['country'].nunique() < 50 or len(g) < 100:
        return None
    G = g.set_index(['country', 'year']).sort_index()
    m2 = PanelOLS(G['_dy'], G[['_dx']], time_effects=True
                  ).fit(cov_type='clustered', cluster_entity=True)
    out['growth'] = float(m2.params['_dx'])
    a = s[s['year'] == y0].set_index('country'); b = s[s['year'] == y1].set_index('country')
    common = a.index.intersection(b.index)
    if len(common) >= 50:
        dy = b.loc[common, 'log_y'] - a.loc[common, 'log_y']
        dx = b.loc[common, 'log_l'] - a.loc[common, 'log_l']
        ml = sm.OLS(dy, sm.add_constant(pd.DataFrame({'d': dx}))).fit(cov_type='HC1')
        out['long_diff'] = float(ml.params['d'])
    return out

results = {}
print(f"{'sensor':>8} {'L':>3} {'#win':>5} {'growth ψ range':>22} {'mean':>7} | {'longdiff range':>20} {'mean':>7}")
for sensor, dd in panels.items():
    lo, hi = int(dd['year'].min()), int(dd['year'].max())
    for L in range(LMIN, LMAX + 1):
        gs, lds, ncs = [], [], []
        for y0 in range(lo, hi - L + 2):
            y1 = y0 + L - 1
            if y1 > hi: continue
            r = growth_and_ld(dd, y0, y1)
            if r is None: continue
            gs.append(r['growth']); ncs.append(r['n_countries'])
            if 'long_diff' in r: lds.append(r['long_diff'])
        if not gs: continue
        key = f'{sensor}_L{L}'
        results[key] = {'n_windows': len(gs), 'growth_min': min(gs), 'growth_max': max(gs),
                        'growth_mean': float(np.mean(gs)),
                        'ld_min': (min(lds) if lds else None), 'ld_max': (max(lds) if lds else None),
                        'ld_mean': (float(np.mean(lds)) if lds else None),
                        'nc_med': int(np.median(ncs))}
        gr = f"[{min(gs):+.3f}, {max(gs):+.3f}]"
        ldr = (f"[{min(lds):+.3f}, {max(lds):+.3f}]" if lds else "--")
        ldm = (f"{np.mean(lds):+.3f}" if lds else "--")
        print(f"{sensor:>8} {L:>3} {len(gs):>5} {gr:>22} {np.mean(gs):+.3f} | {ldr:>20} {ldm:>7}")

# overall: across ALL windows of ALL lengths, the annual growth elasticity bounds
allg = []
for sensor, dd in panels.items():
    lo, hi = int(dd['year'].min()), int(dd['year'].max())
    for L in range(LMIN, LMAX + 1):
        for y0 in range(lo, hi - L + 2):
            y1 = y0 + L - 1
            if y1 > hi: continue
            r = growth_and_ld(dd, y0, y1)
            if r: allg.append(r['growth'])
print(f"\nAcross ALL {len(allg)} windows (L=3..18, both sensors): "
      f"annual growth ψ in [{min(allg):+.3f}, {max(allg):+.3f}], mean {np.mean(allg):+.3f}, "
      f"max |ψ| = {max(abs(min(allg)),abs(max(allg))):.3f}")
results['_summary'] = {'n_windows_total': len(allg), 'growth_min': float(min(allg)),
                       'growth_max': float(max(allg)), 'growth_mean': float(np.mean(allg))}
with open(f'{BASE}/results_window_robustness.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nwritten: results_window_robustness.json')
