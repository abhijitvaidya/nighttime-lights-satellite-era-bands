"""ISO3 keys for GAUL-2015 country names, from data/gaul_country_iso3.csv.

Every GAUL country name in the light reductions has one row: an ISO3 code if it is
joined to a WDI economy, or an exclusion reason (territories, dependencies and
disputed areas without a WDI series; WDI economies that GAUL splits into several
polygons). Joins on ISO3 replace the earlier name normalisation, which dropped the
United States, the United Kingdom and 14 other economies.
"""
from pathlib import Path

import pandas as pd

_XW = pd.read_csv(Path(__file__).resolve().parent / "data" / "gaul_country_iso3.csv",
                  keep_default_na=False)
assert _XW.country.is_unique
GAUL_TO_ISO3 = dict(_XW.loc[_XW.status == "included", ["country", "iso3"]].values)


def gaul_key(names: pd.Series) -> pd.Series:
    """ISO3 for each GAUL country name; missing for excluded or unknown names."""
    unknown = set(names.dropna()) - set(_XW.country)
    if unknown:
        raise ValueError(f"GAUL names missing from the crosswalk: {sorted(unknown)}")
    return names.map(GAUL_TO_ISO3)
