"""Algorithm-correctness gate: reproduce HSW (2012) on THEIR data era.

DMSP stable lights x WDI real GDP, 1992-2008 (HSW's span), ~190 countries.
Published targets:
  - Table 2 col. 1 (ln GDP on ln lights/area, country + year FE): psi = 0.277
  - Table 3 col 4 (long difference, 1992/93 avg -> 2005/06 avg): psi ~ 0.32
  - Growth design (annual dlog with year FE): positive, significant
Conventions copied from HSW: Equatorial Guinea dropped (their stated outlier);
endpoint years averaged in pairs for the long difference.
If these land in range, the pipeline (extraction, merge, name-match, specs)
is validated and era comparisons are meaningful.
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import glob, json, re, unicodedata
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from country_xwalk import gaul_key

BASE = f"{_ROOT}/global_hsw"

# ---------------- WDI GDP ----------------
f = glob.glob(f'{BASE}/data/gdp_constant/API_*.csv')[0]
w = pd.read_csv(f, skiprows=4)
yrs = [c for c in w.columns if c.isdigit()]
gdp = w.melt(id_vars=['Country Name', 'Country Code'], value_vars=yrs,
             var_name='year', value_name='gdp')
gdp['year'] = gdp['year'].astype(int)
meta = pd.read_csv(glob.glob(f'{BASE}/data/gdp_constant/Metadata_Country_*.csv')[0])
gdp = gdp[gdp['Country Code'].isin(set(meta[meta['Region'].notna()]['Country Code']))]

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

gdp['_n'] = gdp['Country Code']

# ---------------- DMSP lights ----------------
dmsp = pd.read_csv(f'{BASE}/data/country_dmsp_annual.csv')
dmsp['_n'] = gaul_key(dmsp['country'])

d = dmsp.merge(gdp[['_n', 'year', 'gdp']], on=['_n', 'year'], how='inner')
d = d[(d['year'] >= 1992) & (d['year'] <= 2008)]
d = d[d['country'] != 'Equatorial Guinea']            # HSW's stated outlier drop
d = d[(d['stable_mean'] > 0) & (d['gdp'] > 0)].copy()
d['log_y'] = np.log(d['gdp'])
d['log_l'] = np.log(d['stable_mean'])
ny = d.groupby('country')['year'].nunique()
d = d[d['country'].isin(ny[ny >= 12].index)]
print(f'HSW-era panel: {len(d)} obs, {d.country.nunique()} countries, '
      f'{d.year.min()}-{d.year.max()}')

R = {}
P = d.set_index(['country', 'year']).sort_index()

# Table-2 analogue: country + year FE, levels
m = PanelOLS(P['log_y'], P[['log_l']], entity_effects=True, time_effects=True
             ).fit(cov_type='clustered', cluster_entity=True)
R['levels_2wfe'] = {'psi': float(m.params['log_l']), 'se': float(m.std_errors['log_l']),
                    'n': int(m.nobs)}

# country FE only (their pooled-FE variant)
m2 = PanelOLS(P['log_y'], P[['log_l']], entity_effects=True
              ).fit(cov_type='clustered', cluster_entity=True)
R['levels_fe'] = {'psi': float(m2.params['log_l']), 'se': float(m2.std_errors['log_l'])}

# HSW Table-3-col-2 analogue: country FE + country-specific linear trend + year FE
# (published value 0.180). Frisch-Waugh: detrend log_y and log_l per country on a
# linear year term, remove year means of the residuals, regress.
def _detrend(gg, col):
    X = sm.add_constant(gg['year'].astype(float))
    return gg[col] - sm.OLS(gg[col], X).fit().predict(X)
dt = d.sort_values(['country', 'year']).copy()
dt['ry'] = dt.groupby('country', group_keys=False).apply(lambda gg: _detrend(gg, 'log_y'))
dt['rx'] = dt.groupby('country', group_keys=False).apply(lambda gg: _detrend(gg, 'log_l'))
dt['ry'] -= dt.groupby('year')['ry'].transform('mean')
dt['rx'] -= dt.groupby('year')['rx'].transform('mean')
mt = sm.OLS(dt['ry'], sm.add_constant(dt['rx'])).fit(
    cov_type='cluster', cov_kwds={'groups': dt['country']})
R['levels_ctrend'] = {'psi': float(mt.params['rx']), 'se': float(mt.bse['rx']),
                      'n': int(mt.nobs)}

# growth design: annual dlog, year FE, consecutive pairs
g = d.sort_values(['country', 'year']).copy()
g['_dy'] = g.groupby('country')['log_y'].diff()
g['_dx'] = g.groupby('country')['log_l'].diff()
g['_gap'] = g.groupby('country')['year'].diff()
g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
G = g.set_index(['country', 'year']).sort_index()
m3 = PanelOLS(G['_dy'], G[['_dx']], time_effects=True
              ).fit(cov_type='clustered', cluster_entity=True)
R['growth_yearfe'] = {'psi': float(m3.params['_dx']), 'se': float(m3.std_errors['_dx']),
                      'n': int(m3.nobs)}

# HSW Table-3-col-4 long difference: avg(1992,1993) -> avg(2005,2006)
def pairmean(years):
    s = d[d['year'].isin(years)].groupby('country')[['log_y', 'log_l']].mean()
    return s
a, b = pairmean([1992, 1993]), pairmean([2005, 2006])
common = a.index.intersection(b.index)
dy = b.loc[common, 'log_y'] - a.loc[common, 'log_y']
dx = b.loc[common, 'log_l'] - a.loc[common, 'log_l']
ml = sm.OLS(dy, sm.add_constant(pd.DataFrame({'d': dx}))).fit(cov_type='HC1')
R['long_diff_9293_0506'] = {'psi': float(ml.params['d']), 'se': float(ml.bse['d']),
                            'n': int(ml.nobs)}

print('\n=== HSW-era validation (targets: 2wFE ~0.28-0.31, long-diff ~0.32) ===')
for k, v in R.items():
    print(f"  {k:<24} psi={v['psi']:+.3f} (se {v['se']:.3f}" +
          (f", n {v['n']})" if 'n' in v else ')'))

verdict = (0.18 <= R['levels_2wfe']['psi'] <= 0.42) and (0.20 <= R['long_diff_9293_0506']['psi'] <= 0.45)
R['verdict_pass'] = bool(verdict)
print(f'\nVALIDATION {"PASS" if verdict else "FAIL"} '
      f'(2wFE {R["levels_2wfe"]["psi"]:.3f}, long-diff {R["long_diff_9293_0506"]["psi"]:.3f})')

with open(f'{BASE}/results_hsw_validation.json', 'w') as fo:
    json.dump(R, fo, indent=2)
print('written: results_hsw_validation.json')
