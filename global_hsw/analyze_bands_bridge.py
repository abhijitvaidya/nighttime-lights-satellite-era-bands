"""Band decomposition (2x2) + cross-sensor bridge at the 2013 overlap.

(A) 2x2 band table: psi (levels 2wFE + growth design) for each VNL band:
        unmasked          masked
  mean    average           average_masked
  median  median            median_masked
  Decomposition: masking effect = masked - unmasked (within statistic);
  statistic effect = median - mean (within masking).

(B) Cross-sensor bridge: 2013 is the only year both DMSP stable_lights and the
  VNL annual composites fly. For the 2013 cross-section, which VNL band best
  reproduces the stable-lights ranking? (log-log Pearson + Spearman, lit
  countries.) The winner is the band that *continues* the DMSP-era series --
  the basis of the paper's recommendation.

Inputs: data/country_ntl_annual.csv (average, median_masked),
        data/country_ntl_bands.csv (average_masked, median),
        data/country_dmsp_annual.csv, WDI gdp folders.
Output: results_bands_bridge.json
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import glob, json, re, unicodedata
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from country_xwalk import gaul_key

BASE = f"{_ROOT}/global_hsw"

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

def attach(df):
    df['_n'] = gaul_key(df['country'])
    return df

# ---------------- lights: merge all four bands ----------------
ntl = attach(pd.read_csv(f'{BASE}/data/country_ntl_annual.csv'))
bands = pd.read_csv(f'{BASE}/data/country_ntl_bands.csv')
ntl = ntl.merge(bands[['gaul_code', 'year', 'avgmask_mean', 'median_mean']],
                on=['gaul_code', 'year'], how='left')

# ---------------- WDI GDP ----------------
f = glob.glob(f'{BASE}/data/gdp_constant/API_*.csv')[0]
w = pd.read_csv(f, skiprows=4)
yrs = [c for c in w.columns if c.isdigit()]
gdp = w.melt(id_vars=['Country Name', 'Country Code'], value_vars=yrs,
             var_name='year', value_name='gdp')
gdp['year'] = gdp['year'].astype(int)
meta = pd.read_csv(glob.glob(f'{BASE}/data/gdp_constant/Metadata_Country_*.csv')[0])
gdp = gdp[gdp['Country Code'].isin(set(meta[meta['Region'].notna()]['Country Code']))]
gdp['_n'] = gdp['Country Code']

d = ntl.merge(gdp[['_n', 'year', 'gdp']], on=['_n', 'year'], how='inner')
d = d[d['gdp'] > 0].copy()
d['log_y'] = np.log(d['gdp'])
ny = d.groupby('country')['year'].nunique()
d = d[d['country'].isin(ny[ny >= 8].index)]

BANDS = [('average', 'avg_mean'), ('average_masked', 'avgmask_mean'),
         ('median', 'median_mean'), ('median_masked', 'medmask_mean')]

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
                            'n': int(m.nobs), 'n_countries': int(s['country'].nunique())},
            'growth_yearfe': {'psi': float(m2.params['_dx']), 'se': float(m2.std_errors['_dx']),
                              'n': int(m2.nobs)}}

results = {'bands': {}}
print('=== 2x2 band table (levels 2wFE | growth yrFE) ===')
for name, col in BANDS:
    if d[col].notna().sum() < 1000:
        print(f'  {name:<16} INSUFFICIENT DATA ({d[col].notna().sum()} non-null)')
        continue
    r = two_specs(d, col)
    results['bands'][name] = r
    print(f"  {name:<16} levels={r['levels_2wfe']['psi']:+.3f} ({r['levels_2wfe']['se']:.3f})"
          f"  growth={r['growth_yearfe']['psi']:+.3f} ({r['growth_yearfe']['se']:.3f})"
          f"  n={r['levels_2wfe']['n']}")

if all(b in results['bands'] for b, _ in BANDS):
    B = {k: results['bands'][k] for k, _ in BANDS}
    for spec in ['levels_2wfe', 'growth_yearfe']:
        mask_mean = B['average_masked'][spec]['psi'] - B['average'][spec]['psi']
        mask_med = B['median_masked'][spec]['psi'] - B['median'][spec]['psi']
        stat_unm = B['median'][spec]['psi'] - B['average'][spec]['psi']
        stat_msk = B['median_masked'][spec]['psi'] - B['average_masked'][spec]['psi']
        results[f'decomposition_{spec}'] = {
            'masking_effect_mean': mask_mean, 'masking_effect_median': mask_med,
            'statistic_effect_unmasked': stat_unm, 'statistic_effect_masked': stat_msk}
        print(f'\n  decomposition [{spec}]: masking {mask_mean:+.3f} (mean) / {mask_med:+.3f} (median); '
              f'statistic {stat_unm:+.3f} (unmasked) / {stat_msk:+.3f} (masked)')

# ---------------- (B) cross-sensor bridge, 2013 ----------------
dmsp = attach(pd.read_csv(f'{BASE}/data/country_dmsp_annual.csv'))
d13 = dmsp[dmsp['year'] == 2013][['gaul_code', 'stable_mean']]
v13 = ntl[ntl['year'] == 2013].merge(d13, on='gaul_code', how='inner')
v13 = v13[v13['stable_mean'] > 0]
print(f'\n=== 2013 cross-sensor bridge ({len(v13)} lit countries) ===')
results['bridge_2013'] = {}
for name, col in BANDS:
    s = v13[v13[col] > 0]
    if len(s) < 50:
        continue
    lp = stats.pearsonr(np.log(s['stable_mean']), np.log(s[col]))[0]
    sp = stats.spearmanr(s['stable_mean'], s[col])[0]
    results['bridge_2013'][name] = {'log_pearson': float(lp), 'spearman': float(sp),
                                    'n': int(len(s))}
    print(f"  {name:<16} log-Pearson={lp:+.3f}  Spearman={sp:+.3f}  n={len(s)}")

with open(f'{BASE}/results_bands_bridge.json', 'w') as fo:
    json.dump(results, fo, indent=2, default=float)
print('\nwritten: results_bands_bridge.json')
