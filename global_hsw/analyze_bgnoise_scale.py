"""Show the bgnoise series approaches EOG's masked band as scale -> 500 m."""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import glob, re, unicodedata, json
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
import warnings; warnings.filterwarnings('ignore')
from country_xwalk import gaul_key

BASE = f"{_ROOT}/global_hsw"
sc = pd.read_csv(f'{BASE}/data/country_bgnoise_scale.csv')
bands = pd.read_csv(f'{BASE}/data/country_ntl_annual.csv')[['country', 'year', 'medmask_mean']]

def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z]', '', s)
sc['_n'] = gaul_key(sc['country']); bands['_n'] = gaul_key(bands['country'])
sc = sc.merge(bands[['_n', 'year', 'medmask_mean']], on=['_n', 'year'], how='left')

f = glob.glob(f'{BASE}/data/gdp_constant/API_*.csv')[0]
w = pd.read_csv(f, skiprows=4); yrs = [c for c in w.columns if c.isdigit()]
gdp = w.melt(id_vars=['Country Name', 'Country Code'], value_vars=yrs, var_name='year', value_name='gdp')
gdp['year'] = gdp['year'].astype(int); gdp['_n'] = gdp['Country Code']
sc = sc.merge(gdp[['_n', 'year', 'gdp']], on=['_n', 'year'], how='inner')
sc = sc[sc['gdp'] > 0].copy(); sc['log_y'] = np.log(sc['gdp'])

def growth_corr(df, a, b):
    cs = []
    for c, g in df.sort_values('year').groupby('country'):
        x = np.log(g[a].replace({0: np.nan})).diff(); y = np.log(g[b].replace({0: np.nan})).diff()
        v = x.notna() & y.notna()
        if v.sum() >= 6 and x[v].std() > 0 and y[v].std() > 0:
            cs.append(np.corrcoef(x[v], y[v])[0, 1])
    return np.array(cs)

def panel_growth(df, xcol):
    s = df[df[xcol] > 0].dropna(subset=[xcol]).copy(); s['log_l'] = np.log(s[xcol])
    g = s.sort_values(['country', 'year']).copy()
    g['_dy'] = g.groupby('country')['log_y'].diff(); g['_dx'] = g.groupby('country')['log_l'].diff()
    g['_gap'] = g.groupby('country')['year'].diff(); g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
    G = g.set_index(['country', 'year']).sort_index()
    m = PanelOLS(G['_dy'], G[['_dx']], time_effects=True).fit(cov_type='clustered', cluster_entity=True)
    return float(m.params['_dx']), float(g['_dx'].std())

out = {}
print(f"n countries: {sc.country.nunique()}  ({sorted(sc.country.unique())})")
print(f"\n{'scale':>7} {'stat vs medmask growth corr (med)':>34} {'stat growth psi':>16} {'sd(dlog stat)':>14} {'sd(dlog raw)':>13}")
for s in sorted(sc['scale'].unique(), reverse=True):
    d = sc[sc['scale'] == s]
    cc = growth_corr(d, 'stat', 'medmask_mean')
    psi, sdstat = panel_growth(d, 'stat')
    _, sdraw = panel_growth(d, 'raw')
    rawcc = growth_corr(d, 'raw', 'medmask_mean')
    out[int(s)] = {'stat_corr_med': float(np.median(cc)), 'raw_corr_med': float(np.median(rawcc)),
                   'stat_growth_psi': psi, 'sd_stat': sdstat, 'sd_raw': sdraw, 'n_countries': int(d.country.nunique())}
    print(f"{s:>6}m {np.median(cc):>+33.2f} {psi:>+16.3f} {sdstat:>14.3f} {sdraw:>13.3f}")
# EOG medmask reference on this panel
mm_psi, mm_sd = panel_growth(sc[sc['scale'] == 500], 'medmask_mean')
print(f"\nEOG medmask (this panel): growth psi {mm_psi:+.3f}, sd {mm_sd:.3f}")
out['medmask_ref'] = {'growth_psi': mm_psi, 'sd': mm_sd}
json.dump(out, open(f'{BASE}/results_bgnoise_scale.json', 'w'), indent=2)
print('\nwritten: results_bgnoise_scale.json')
