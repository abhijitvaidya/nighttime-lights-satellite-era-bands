"""Figures 1-3 of the paper, from frozen JSON outputs only.

Writes figures/p0_fig1_rolling.png, p0_fig2_validation.png and
p0_fig3_bands_common.png (the manuscript's Figures 2, 1 and 3).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from p0_config import VIIRS_START  # noqa: E402
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG = BASE.parent / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})
BLUE, RED, GREEN, GREY = '#1f4e79', '#c0504d', '#9bbb59', '#7f7f7f'

# ---------------- Fig 1: rolling-window psi time path ----------------
R = json.load(open(f'{BASE}/results_rolling.json'))
def series(sensor, spec):
    pts = []
    for k, v in R.items():
        if k.startswith(sensor + '_') and spec in v:
            y0, y1 = int(k.split('_')[1]), int(k.split('_')[2])
            pts.append((y1, v[spec]['psi'], v[spec]['se']))
    return sorted(pts)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharex=True)
for ax, spec, title in [(axes[0], 'level_2wfe', 'Country and year effects'),
                        (axes[1], 'growth_yearfe', 'Annual changes and year effects')]:
    d = series('DMSP', spec)
    xs, ys, es = [p[0] for p in d], [p[1] for p in d], [p[2] for p in d]
    ax.errorbar(xs, ys, yerr=[1.96*e for e in es], fmt='o-', color=BLUE,
                capsize=2, ms=4, label='DMSP stable lights')
    for sens, col, lab in [('VIIRSmm', GREEN, 'VIIRS median-masked'),
                           ('VIIRSav', RED, 'VIIRS unmasked average')]:
        v = series(sens, spec)
        if v:
            # every VIIRS window in the main panel
            ax.errorbar([q[0] for q in v], [q[1] for q in v], yerr=[1.96*q[2] for q in v],
                        fmt='s-', color=col, capsize=3, ms=6, label=lab)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel('window end year (12-year windows)')
axes[0].axhline(0.30, color=RED, lw=0.8, ls='--')
axes[0].text(2015, 0.305, 'HSW 0.30', color=RED, fontsize=7.5)
axes[0].set_ylabel('GDP-on-radiance coefficient (95% CI)')
axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
fig.suptitle('GDP-on-radiance coefficients over 12-year windows, 1992–2025',
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(f'{FIG}/p0_fig1_rolling.png', bbox_inches='tight'); plt.close(fig)
print('p0_fig1 done')

# ---------------- Fig 2: HSW-era validation forest ----------------
V = json.load(open(f'{BASE}/results_hsw_validation.json'))
P = json.load(open(f'{BASE}/results_pilot.json'))
mm = P['median_masked']
rows = [
    ('1992–2008 DMSP (HSW era, this paper)', 'country and year effects', V['levels_2wfe']['psi'], V['levels_2wfe']['se']),
    ('1992–2008 DMSP (HSW era, this paper)', 'long difference, 1992/93 to 2005/06', V['long_diff_9293_0506']['psi'], V['long_diff_9293_0506']['se']),
    (f'{VIIRS_START}–2025 VIIRS median-masked', 'country and year effects', mm['country_year_fe']['psi'], mm['country_year_fe']['se']),
    (f'{VIIRS_START}–2025 VIIRS median-masked', f'long difference, {VIIRS_START} to 2025', mm['long_diff']['psi'], mm['long_diff']['se']),
    (f'{VIIRS_START}–2025 VIIRS median-masked', 'annual changes and year effects', mm['henderson_growth_yearfe']['psi'], mm['henderson_growth_yearfe']['se']),
    ('1992–2008 DMSP (HSW era, this paper)', 'annual changes and year effects', V['growth_yearfe']['psi'], V['growth_yearfe']['se']),
]
fig, ax = plt.subplots(figsize=(7.2, 4))
ys = np.arange(len(rows))[::-1]
for y, (era, spec, psi, se) in zip(ys, rows):
    col = BLUE if 'DMSP' in era else GREEN
    ax.errorbar(psi, y, xerr=1.96*se, fmt='o', color=col, capsize=3, ms=5)
ax.axvline(0.30, color=RED, lw=0.9, ls='--')
ax.text(0.30, 0.98, 'HSW 0.30', color=RED, fontsize=7.5,
        transform=ax.get_xaxis_transform(), ha='left', va='top')
ax.axvline(0, color=GREY, lw=0.7)
ax.set_yticks(ys)
ax.set_yticklabels([f'{e.split("(")[0].strip()}: {s}' for e, s, _, _ in rows], fontsize=8)
ax.set_xlabel('GDP-on-radiance coefficient (95% CI)')
ax.set_title('DMSP validation and VIIRS comparison', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/p0_fig2_validation.png'); plt.close(fig)
print('p0_fig2 done')

# ---------------- Fig 3: four bands on identical observations ----------------
C = json.load(open(f'{BASE}/results_p0_common_sample.json'))['band_estimates']
order = [('average', 'Average'), ('median', 'Median'), ('average_masked', 'Average-masked'), ('median_masked', 'Median-masked')]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (bk, sk, title) in zip(axes, [('panel_beta', 'panel_se', 'Country and year effects'),
                                     ('annual_beta', 'annual_se', 'Annual changes and year effects')]):
    ys = list(range(len(order)))[::-1]
    for y, (key, label) in zip(ys, order):
        col = '#1f5fa8' if 'masked' in key else '#8c8c8c'
        ax.errorbar(C[key][bk], y, xerr=1.96 * C[key][sk], fmt='o', color=col, capsize=3, lw=1.5)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([l for _, l in order])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('GDP-on-radiance coefficient (95% CI)')
fig.suptitle(f'Four VIIRS annual bands on identical observations, {VIIRS_START}–2025', fontsize=11)
fig.tight_layout()
fig.savefig(f'{FIG}/p0_fig3_bands_common.png'); plt.close(fig)
print('p0_fig3 done')
print('figures written to figures/')
