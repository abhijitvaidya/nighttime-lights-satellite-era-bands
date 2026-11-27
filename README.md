# Nighttime Lights and Economic Growth across Satellite Eras: Band Choice and Persistence

Replication package for the working paper of the same title by Abhijit Vaidya (Working Paper, Version 1, November 2026).

- Repository: <https://github.com/abhijitvaidya/nighttime-lights-satellite-era-bands>
- Archived release: <https://doi.org/10.5281/zenodo.24581893>
- Working paper: link to be added when the paper is posted

The paper compares the four native annual VIIRS nighttime-lights bands (average, median, average-masked and median-masked) as measures of country GDP for 202 economies over 2014–2025, on identical observations and in retrospective held-out prediction, and places the VIIRS estimates beside DMSP estimates for the period studied by Henderson, Storeygard and Weil (2012). Background-masked bands explain more of the cross-country variation in annual GDP changes than unmasked bands, mainly in dimmer and poorer economies, and in both sensor eras the GDP-on-lights coefficient rises with the differencing horizon.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
python run_all.py --verify   # check every frozen output against OUTPUT_MANIFEST.csv
python run_all.py            # rerun all 20 estimation and figure steps (about a minute)
```

Tested with Python 3.13 on macOS. All inputs are public and bundled; `DATA_SOURCES.md` gives each input's provider, version, access date and licence, and `INPUT_MANIFEST.csv` its size and checksum. The analysis does not need Earth Engine. Re-extracting the country values from the satellite images needs a Google Earth Engine account: set `EE_PROJECT` to your Cloud project before running any `global_hsw/extract_*.py` script.

## Contents

- `run_all.py`: runs the estimation steps in order; `--verify` checks the frozen outputs.
- `global_hsw/`: extraction scripts (`extract_*.py`), estimation scripts (`validate_hsw.py`, `analyze_*.py`), figure script, shared helpers (`p0_config.py`, `country_xwalk.py`, `gaul_tiles.py`) and frozen results (`results_*.json`).
- `global_hsw/data/`: bundled inputs and derived panels.
- `figures/`: Figures 1 to 3 of the paper.
- `nighttime_lights.py`: monthly cleaning functions used by `analyze_pstt_country.py`.

## Where each result comes from

Scripts are in `global_hsw/`, listed in the order `run_all.py` runs them.

| Script | Estimates | Reported in |
|---|---|---|
| `validate_hsw.py` | DMSP validation, 1992–2008 | Table 1, Figure 1 |
| `analyze_p0_dmsp_satellite.py` | DMSP validation with satellite-specific data | Section 5.1, Appendix P |
| `analyze_global_pilot.py` | Estimation panel (`data/global_panel.csv`) and VIIRS specifications | Table 2, Figure 1, Table C2 |
| `analyze_rolling_windows.py` | Rolling 12-year windows | Section 5.1, Figure 2 |
| `analyze_window_robustness.py` | Window lengths of three to eighteen years | Section 5.1, Table C1 |
| `analyze_bands_bridge.py` | Four bands on band-specific samples; 2013 sensor bridge | Supporting output |
| `analyze_pstt_country.py` | Temporal cleaning of the monthly series | Section 6 |
| `analyze_bgnoise_country.py` | Pixel-threshold mask, 48 countries | Appendices D and H |
| `analyze_bgnoise_scale.py` | Threshold mask at several resolutions | Appendix D |
| `analyze_p0_common_sample.py` | Four bands and threshold variants on identical observations | Table 3, Figure 3, Appendices G and H |
| `analyze_p0_oos.py` | Held-out prediction, paired comparisons and forecast tests | Tables 4 and 5, Appendices E and F |
| `analyze_p0_heterogeneity.py` | Masking gain by brightness, income and statistical capacity | Table 6, Appendix K |
| `analyze_p0_dmsp_heldout.py` | Held-out prediction on DMSP stable lights | Section 5.3, Appendix L |
| `analyze_p0_vanuatu_sensitivity.py` | Estimates with Vanuatu included; Leinonen's specifications | Section 2, Appendix M |
| `analyze_p0_overlap_2012.py` | The 2012 and 2013 sensor overlap | Section 6, Appendices M and N |
| `analyze_p0_fiscal_year.py` | Calendar-year lights against fiscal-year GDP | Section 3, Appendix O |
| `analyze_p0_permutation.py` | Whole-history reassignment | Section 5.3, Appendix J |
| `analyze_p0_descriptives.py` | Descriptive statistics | Section 3, Table A1 |
| `extract_p0_band_resolution.py` | Four bands at 1 km and 2 km, from a cached reduction | Appendix D |
| `make_paper_figures.py` | Figures 1 to 3 | Figures 1 to 3 |

`OUTPUT_MANIFEST.csv` lists the frozen outputs; Appendix B of the paper's supplement also lists the extraction scripts and their outputs.

## Licence and citation

Code is MIT (`LICENSE`). Figures are CC BY 4.0. Bundled data keep their providers' licences (CC BY 4.0 for EOG VIIRS, World Bank WDI and Penn World Table; public domain for NOAA DMSP). Please cite the working paper; `CITATION.cff` gives the reference.
