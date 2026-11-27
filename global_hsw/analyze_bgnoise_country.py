"""Does pixel-level bgnoise BEFORE aggregation recover the masked-band elasticity?

Puts six radiance series on the SAME country panel and runs identical specs:
  EOG_avg      EOG annual 'average' band (unmasked)        -- country mean
  EOG_medmask  EOG annual 'median_masked' band             -- country mean
  raw          my monthly cf-weighted annual mean, no mask -- the average analogue
  lit          bgnoise per-year lit mask, lit-pixel mean
  zero         bgnoise per-year, dark pixels set to 0       -- the EOG-mask analogue
  stat         bgnoise with a single trailing mask from the final year (2025)

If raw tracks EOG_avg (low psi, high noise) while zero/lit/stat track EOG_medmask
(higher psi, low noise), then pixel-level background suppression before aggregation
is the operative ingredient, whether it comes from EOG's producer mask or PSTT's
own bgnoise. The within-panel raw-vs-masked contrast is the clean test; absolute
levels need not match the 188-country headline because n is small by design.

Output: results_bgnoise_country.json + console table.
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import glob, json, re, unicodedata
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import warnings
from country_xwalk import gaul_key
warnings.filterwarnings('ignore')

BASE = f"{_ROOT}/global_hsw"

bg = pd.read_csv(f'{BASE}/data/country_bgnoise_annual.csv')
bands = pd.read_csv(f'{BASE}/data/country_ntl_annual.csv')[['country', 'year', 'avg_mean', 'medmask_mean']]

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

def keyed(df):
    df = df.copy()
    df['_n'] = gaul_key(df['country'])
    return df

bg = keyed(bg); bands = keyed(bands)
d = bg.merge(bands[['_n', 'year', 'avg_mean', 'medmask_mean']], on=['_n', 'year'], how='left')

# WDI GDP
f = glob.glob(f'{BASE}/data/gdp_constant/API_*.csv')[0]
w = pd.read_csv(f, skiprows=4)
yrs = [c for c in w.columns if c.isdigit()]
gdp = w.melt(id_vars=['Country Name', 'Country Code'], value_vars=yrs, var_name='year', value_name='gdp')
gdp['year'] = gdp['year'].astype(int)
meta = pd.read_csv(glob.glob(f'{BASE}/data/gdp_constant/Metadata_Country_*.csv')[0])
gdp = gdp[gdp['Country Code'].isin(set(meta[meta['Region'].notna()]['Country Code']))]
gdp['_n'] = gdp['Country Code']

d = d.merge(gdp[['_n', 'year', 'gdp']], on=['_n', 'year'], how='inner')
d = d[d['gdp'] > 0].copy()
d['log_y'] = np.log(d['gdp'])
print(f'panel: {len(d)} country-years, {d.country.nunique()} countries, {d.year.min()}-{d.year.max()}')
print('countries:', sorted(d['country'].unique()))

def two_specs(dd, xcol):
    s = dd[dd[xcol] > 0].dropna(subset=[xcol]).copy()
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
    m2 = PanelOLS(G['_dy'], G[['_dx']], time_effects=True).fit(cov_type='clustered', cluster_entity=True)
    return {'levels_2wfe': {'psi': float(m.params['log_l']), 'se': float(m.std_errors['log_l']), 'n': int(m.nobs)},
            'growth_yearfe': {'psi': float(m2.params['_dx']), 'se': float(m2.std_errors['_dx']), 'n': int(m2.nobs)},
            'dlogl_sd': float(g['_dx'].std())}

SERIES = [('EOG_avg', 'avg_mean'), ('EOG_medmask', 'medmask_mean'),
          ('raw', 'raw'), ('lit', 'lit'), ('zero', 'zero'), ('stat', 'stat')]
results = {}
print(f"\n{'series':<14} {'levels 2wFE':>18} {'growth yrFE':>18} {'sd(dlogL)':>10}")
for name, col in SERIES:
    try:
        r = two_specs(d, col); results[name] = r
        print(f"{name:<14} {r['levels_2wfe']['psi']:+.3f} ({r['levels_2wfe']['se']:.3f})"
              f"   {r['growth_yearfe']['psi']:+.3f} ({r['growth_yearfe']['se']:.3f})   {r['dlogl_sd']:>9.3f}")
    except Exception as e:
        print(f"{name:<14} FAIL {type(e).__name__}: {e}")

# Within-country agreement of the fixed-mask series with the producer median-masked band
from scipy.stats import spearmanr
agree = []
for name, g in bg.merge(bands[['_n', 'year', 'medmask_mean']], on=['_n', 'year'], how='left').groupby('country'):
    g = g.sort_values('year')
    g = g[(g['stat'] > 0) & (g['medmask_mean'] > 0)]
    if len(g) < 5:
        continue
    ls, lm = np.log(g['stat']), np.log(g['medmask_mean'])
    agree.append({'country': name, 'scale_m': int(g['scale_m'].iloc[0]),
                  'level_corr': float(np.corrcoef(ls, lm)[0, 1]),
                  'growth_corr': float(np.corrcoef(ls.diff().dropna(), lm.diff().dropna())[0, 1])})
agree = pd.DataFrame(agree)
results['stat_vs_medmask_agreement'] = {
    'n_countries': int(len(agree)),
    'median_level_corr': float(agree['level_corr'].median()),
    'median_growth_corr': float(agree['growth_corr'].median()),
    'median_growth_corr_by_scale': {int(k): float(v) for k, v in agree.groupby('scale_m')['growth_corr'].median().items()},
    'countries_by_scale': {int(k): int(v) for k, v in agree.groupby('scale_m').size().items()},
    'spearman_scale_vs_growth_corr': float(spearmanr(agree['scale_m'], agree['growth_corr']).statistic),
    'growth_corr_by_country': dict(zip(agree['country'], agree['growth_corr'].round(4))),
}

results['_meta'] = {'n_countries': int(d.country.nunique()), 'n_obs': int(len(d)),
                    'countries': sorted(d['country'].unique())}
json.dump(results, open(f'{BASE}/results_bgnoise_country.json', 'w'), indent=2)
print('\nwritten: results_bgnoise_country.json')
