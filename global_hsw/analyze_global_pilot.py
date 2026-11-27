"""Pilot: HSW (2012) specification battery on the VIIRS-era global panel.

Panel: country-level annual VNL lights (VIIRS_START-2025, p0_config.py) x WDI real GDP (constant
2015 US$, through 2025). HSW's convention: regress log GDP on log lights
(mean radiance = their lights/area), country FE + year FE as the headline.
Specifications follow HSW's estimators (levels with country and year effects, annual
changes, long difference).

Outputs: data/global_panel.csv, results_pilot.json, console summary.
"""
import os as _os_root
_ROOT = _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__)))
import json, glob, re, unicodedata
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from country_xwalk import gaul_key
from p0_config import VIIRS_START

rng = np.random.default_rng(42)
BASE = f"{_ROOT}/global_hsw"

# ---------------------------------------------------------------- lights
ntl = pd.read_csv(f'{BASE}/data/country_ntl_annual.csv')

# ---------------------------------------------------------------- WDI
def load_wdi(folder, name):
    f = glob.glob(f'{BASE}/data/{folder}/API_*.csv')[0]
    w = pd.read_csv(f, skiprows=4)
    yrs = [c for c in w.columns if c.isdigit()]
    long = w.melt(id_vars=['Country Name', 'Country Code'], value_vars=yrs,
                  var_name='year', value_name=name)
    long['year'] = long['year'].astype(int)
    return long.rename(columns={'Country Name': 'wb_name', 'Country Code': 'iso3'})

gdp = load_wdi('gdp_constant', 'gdp')
pc = load_wdi('gdppc_constant', 'gdppc')
wdi = gdp.merge(pc, on=['wb_name', 'iso3', 'year'])

# drop WDI aggregates (regions, income groups) — aggregates have iso3 codes
# not attached to a single economy; identify via the metadata file's Region field
meta = pd.read_csv(glob.glob(f'{BASE}/data/gdp_constant/Metadata_Country_*.csv')[0])
real = set(meta[meta['Region'].notna()]['Country Code'])
wdi = wdi[wdi['iso3'].isin(real)]

# ---------------------------------------------------------------- country join
# ISO3 join through the explicit GAUL crosswalk (data/gaul_country_iso3.csv), which
# lists every GAUL name with its ISO3 code or the reason it is excluded.
ntl['_n'] = gaul_key(ntl['country'])
wdi['_n'] = wdi['iso3']
panel = ntl.merge(wdi, on=['_n', 'year'], how='inner')
excluded = sorted(set(ntl.loc[ntl['_n'].isna(), 'country']))
no_wdi = sorted(set(ntl.loc[ntl['_n'].notna() & ~ntl['_n'].isin(set(wdi['_n'])), 'country']))
print(f'NTL countries: {ntl["country"].nunique()} | matched into panel: '
      f'{panel["country"].nunique()} | excluded by crosswalk: {len(excluded)} | '
      f'crosswalk ISO3 without WDI rows: {no_wdi}')

panel = panel.dropna(subset=['gdp'])
panel_all = panel.copy()  # includes 2013 (no stray-light correction); used only for a robustness row
panel = panel[panel['year'] >= VIIRS_START]
panel = panel[(panel['avg_mean'] > 0) & (panel['gdp'] > 0)]
panel['log_y'] = np.log(panel['gdp'])
panel['log_l_avg'] = np.log(panel['avg_mean'])
med_ok = panel['medmask_mean'] > 0
panel['log_l_med'] = np.where(med_ok, np.log(panel['medmask_mean']), np.nan)

# sample rule: >= 8 paired years of lights and GDP (the panel is nearly balanced)
ny = panel.groupby('country')['year'].nunique()
panel = panel[panel['country'].isin(ny[ny >= 8].index)].copy()
panel_all = panel_all[panel_all['country'].isin(set(panel['country']))
                      & (panel_all['avg_mean'] > 0) & (panel_all['gdp'] > 0)].copy()
for c, x in (('log_y', 'gdp'), ('log_l_avg', 'avg_mean')):
    panel_all[c] = np.log(panel_all[x])
panel_all['log_l_med'] = np.log(panel_all['medmask_mean'].where(panel_all['medmask_mean'] > 0))

# initial-period income (mean GDP per capita over the first two panel years), time-invariant,
# used as a moderator that does not vary within country
pc0 = (panel[panel['year'] <= VIIRS_START + 1].groupby('country')['gdppc'].mean()
       .rename('pc0'))
panel = panel.merge(pc0, on='country', how='left')
panel['log_pc0_dm'] = np.log(panel['pc0']) - np.log(panel['pc0']).mean()

panel.to_csv(f'{BASE}/data/global_panel.csv', index=False)
print(f'Panel: {len(panel)} country-years, {panel["country"].nunique()} countries, '
      f'{panel["year"].min()}-{panel["year"].max()}')

# ---------------------------------------------------------------- specs
def coefs(m, x, extra=None):
    out = {'psi': float(m.params[x]), 'se': float(m.std_errors[x]),
           'p': float(m.pvalues[x]), 'n': int(m.nobs)}
    for k, v in (extra or {}).items():
        out[k] = {'coef': float(m.params[v]), 'se': float(m.std_errors[v]),
                  'p': float(m.pvalues[v])}
    return out

def battery(d, xcol, tag):
    d = d.dropna(subset=[xcol]).copy()
    P = d.set_index(['country', 'year']).sort_index()
    Pt = P.copy(); Pt['_trend'] = Pt.index.get_level_values('year') - VIIRS_START
    R = {}
    X = sm.add_constant(P[[xcol]])
    m = sm.OLS(P['log_y'], X).fit(cov_type='cluster',
        cov_kwds={'groups': P.index.get_level_values('country')})
    R['pooled_ols'] = {'psi': float(m.params[xcol]), 'se': float(m.bse[xcol]),
                       'p': float(m.pvalues[xcol]), 'n': int(m.nobs)}
    R['country_fe'] = coefs(PanelOLS(P['log_y'], P[[xcol]], entity_effects=True)
                            .fit(cov_type='clustered', cluster_entity=True), xcol)
    R['country_fe_trend'] = coefs(PanelOLS(Pt['log_y'], Pt[[xcol, '_trend']],
                                           entity_effects=True)
                                  .fit(cov_type='clustered', cluster_entity=True), xcol)
    R['country_year_fe'] = coefs(PanelOLS(P['log_y'], P[[xcol]], entity_effects=True,
                                          time_effects=True)
                                 .fit(cov_type='clustered', cluster_entity=True), xcol)
    # HSW Table-3-col-2 analogue, as in validate_hsw.py: country effects, country-specific linear
    # trends and year effects. Detrend log GDP and log lights per country on a linear year term,
    # remove year means of the residuals, regress.
    def _detrend(gg, col):
        Xt = sm.add_constant(gg['year'].astype(float))
        return gg[col] - sm.OLS(gg[col], Xt).fit().predict(Xt)
    dt = d.sort_values(['country', 'year']).copy()
    dt['ry'] = dt.groupby('country', group_keys=False).apply(lambda gg: _detrend(gg, 'log_y'))
    dt['rx'] = dt.groupby('country', group_keys=False).apply(lambda gg: _detrend(gg, xcol))
    dt['ry'] -= dt.groupby('year')['ry'].transform('mean')
    dt['rx'] -= dt.groupby('year')['rx'].transform('mean')
    mt = sm.OLS(dt['ry'], sm.add_constant(dt['rx'])).fit(
        cov_type='cluster', cov_kwds={'groups': dt['country']})
    R['country_trends_year_fe'] = {'psi': float(mt.params['rx']), 'se': float(mt.bse['rx']),
                                   'p': float(mt.pvalues['rx']), 'n': int(mt.nobs)}
    # HSW growth design: annual log differences, year FE, consecutive years
    g = d.sort_values(['country', 'year']).copy()
    g['_dy'] = g.groupby('country')['log_y'].diff()
    g['_dx'] = g.groupby('country')[xcol].diff()
    g['_gap'] = g.groupby('country')['year'].diff()
    g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
    G = g.set_index(['country', 'year']).sort_index()
    R['henderson_growth_yearfe'] = coefs(
        PanelOLS(G['_dy'], G[['_dx']], time_effects=True)
        .fit(cov_type='clustered', cluster_entity=True), '_dx')
    R['henderson_growth_yearfe']['n_units'] = int(g['country'].nunique())
    # long difference, fixed window first->last common years
    y0, y1 = int(d['year'].min()), int(d['year'].max())
    a = d[d['year'] == y0].set_index('country')
    b = d[d['year'] == y1].set_index('country')
    common = a.index.intersection(b.index)
    dy = b.loc[common, 'log_y'] - a.loc[common, 'log_y']
    dx = b.loc[common, xcol] - a.loc[common, xcol]
    ml = sm.OLS(dy, sm.add_constant(pd.DataFrame({'d': dx}))).fit(cov_type='HC1')
    R['long_diff'] = {'psi': float(ml.params['d']), 'se': float(ml.bse['d']),
                      'p': float(ml.pvalues['d']), 'n': int(ml.nobs),
                      'window': f'{y0}-{y1}'}
    # income-gradient interaction (initial income, FE + trend)
    Di = Pt.dropna(subset=['log_pc0_dm']).copy()
    Di['_int'] = Di[xcol] * Di['log_pc0_dm']
    R['fe_trend_initialpc_interaction'] = coefs(
        PanelOLS(Di['log_y'], Di[[xcol, '_int', '_trend']], entity_effects=True)
        .fit(cov_type='clustered', cluster_entity=True), xcol, {'interaction': '_int'})
    # block-permutation placebo on FE and growth design
    wide = d.pivot_table(index='country', columns='year', values=xcol)
    ents = wide.index.to_numpy()
    blk_fe, blk_gr = [], []
    for _ in range(200):
        perm = rng.permutation(len(ents))
        lng = (wide.set_axis(ents[perm], axis=0).stack()
               .rename('xp').reset_index())
        lng.columns = ['country', 'year', 'xp']
        mm = d[['country', 'year', 'log_y']].merge(lng, on=['country', 'year']).dropna()
        Mp = mm.set_index(['country', 'year']).sort_index()
        try:
            blk_fe.append(float(PanelOLS(Mp['log_y'], Mp[['xp']], entity_effects=True)
                                .fit().params['xp']))
        except Exception:
            pass
        gg = mm.sort_values(['country', 'year']).copy()
        gg['_dy'] = gg.groupby('country')['log_y'].diff()
        gg['_dx'] = gg.groupby('country')['xp'].diff()
        gg = gg.dropna(subset=['_dy', '_dx'])
        Gp = gg.set_index(['country', 'year']).sort_index()
        try:
            blk_gr.append(float(PanelOLS(Gp['_dy'], Gp[['_dx']], time_effects=True)
                                .fit().params['_dx']))
        except Exception:
            pass
    R['placebo_block_fe'] = {'mean': float(np.mean(blk_fe)),
                             'ci95': [float(np.percentile(blk_fe, 2.5)),
                                      float(np.percentile(blk_fe, 97.5))]}
    R['placebo_block_growth'] = {'mean': float(np.mean(blk_gr)),
                                 'ci95': [float(np.percentile(blk_gr, 2.5)),
                                          float(np.percentile(blk_gr, 97.5))]}
    print(f'\n=== {tag} ===')
    for k in ['pooled_ols', 'country_fe', 'country_fe_trend', 'country_year_fe', 'country_trends_year_fe',
              'henderson_growth_yearfe', 'long_diff', 'fe_trend_initialpc_interaction']:
        r = R[k]
        extra = ''
        if 'interaction' in r:
            extra = f"  int={r['interaction']['coef']:+.3f} (p={r['interaction']['p']:.2g})"
        if 'window' in r:
            extra = f"  [{r['window']}]"
        print(f"  {k:<34} psi={r['psi']:+.3f} (se {r['se']:.3f}, p {r['p']:.3g}, n {r['n']}){extra}")
    for k in ['placebo_block_fe', 'placebo_block_growth']:
        r = R[k]
        print(f"  {k:<34} mean={r['mean']:+.3f} 95% [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    return R

results = {'meta': {'countries': int(panel['country'].nunique()), 'obs': len(panel)}}
results['avg'] = battery(panel, 'log_l_avg', 'VNL average radiance (HSW lights/area analogue)')
results['median_masked'] = battery(panel, 'log_l_med', 'VNL median_masked (stable-lights analogue)')

# ---------------- robustness battery (median_masked) ----------------
def two_specs(d, xcol):
    """levels 2wFE + growth design, the two summary statistics."""
    d = d.dropna(subset=[xcol]).copy()
    P = d.set_index(['country', 'year']).sort_index()
    m = PanelOLS(P['log_y'], P[[xcol]], entity_effects=True, time_effects=True
                 ).fit(cov_type='clustered', cluster_entity=True)
    g = d.sort_values(['country', 'year']).copy()
    g['_dy'] = g.groupby('country')['log_y'].diff()
    g['_dx'] = g.groupby('country')[xcol].diff()
    g['_gap'] = g.groupby('country')['year'].diff()
    g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
    G = g.set_index(['country', 'year']).sort_index()
    m2 = PanelOLS(G['_dy'], G[['_dx']], time_effects=True
                  ).fit(cov_type='clustered', cluster_entity=True)
    return {'levels_2wfe': {'psi': float(m.params[xcol]), 'se': float(m.std_errors[xcol]),
                            'n': int(m.nobs)},
            'growth_yearfe': {'psi': float(m2.params[xcol.replace(xcol, '_dx')]),
                              'se': float(m2.std_errors['_dx']), 'n': int(m2.nobs)}}

rob = {}
rob['covid_excl'] = two_specs(panel[~panel['year'].isin([2020, 2021])], 'log_l_med')
rob['v21_only'] = two_specs(panel[panel['year'] <= 2021], 'log_l_med')
rob['with_2013'] = two_specs(panel_all, 'log_l_med')  # adds the 2013 composite, built without stray-light correction
# PWT 11.0 real GDP (rgdpna), available through 2023
pwt = pd.read_excel(f'{BASE}/data/pwt110.xlsx', sheet_name='Data')[
    ['countrycode', 'year', 'rgdpna']].rename(columns={'countrycode': 'iso3'})
pp = panel.merge(pwt, on=['iso3', 'year'], how='inner').dropna(subset=['rgdpna'])
pp = pp[pp['rgdpna'] > 0].copy()
pp['log_y'] = np.log(pp['rgdpna'])
if len(pp) > 500:
    rob['pwt_gdp'] = two_specs(pp, 'log_l_med')
results['robustness_median_masked'] = rob


def winsorised_annual(d, xcol, q=0.01):
    """Annual-change slope (year effects) after winsorising both changes at q and 1-q."""
    g = d.dropna(subset=[xcol]).sort_values(['country', 'year']).copy()
    g['_dy'] = g.groupby('country')['log_y'].diff()
    g['_dx'] = g.groupby('country')[xcol].diff()
    g['_gap'] = g.groupby('country')['year'].diff()
    g = g[g['_gap'] == 1].dropna(subset=['_dy', '_dx'])
    sd_raw = float(g['_dx'].std())
    for c in ['_dx', '_dy']:
        lo, hi = g[c].quantile([q, 1 - q])
        g[c] = g[c].clip(lo, hi)
    G = g.set_index(['country', 'year']).sort_index()
    m = PanelOLS(G['_dy'], G[['_dx']], time_effects=True).fit(cov_type='clustered', cluster_entity=True)
    return {'sd_annual_log_radiance_change': sd_raw,
            'winsorised_growth_yearfe': {'psi': float(m.params['_dx']), 'se': float(m.std_errors['_dx']),
                                         'n': int(m.nobs), 'quantiles': [q, 1 - q]}}


results['annual_change_dispersion'] = {'median_masked': winsorised_annual(panel, 'log_l_med'),
                                       'avg': winsorised_annual(panel, 'log_l_avg')}
print('\n=== robustness (median_masked) ===')
for k, v in rob.items():
    print(f"  {k:<22} 2wFE={v['levels_2wfe']['psi']:+.3f} ({v['levels_2wfe']['se']:.3f})"
          f"  growth={v['growth_yearfe']['psi']:+.3f} ({v['growth_yearfe']['se']:.3f})"
          f"  n={v['levels_2wfe']['n']}")

with open(f'{BASE}/results_pilot.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
print('\nwritten: results_pilot.json')
