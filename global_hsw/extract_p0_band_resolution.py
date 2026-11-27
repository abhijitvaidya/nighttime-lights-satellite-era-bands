"""P0 sensitivity of the four producer bands to reduction scale.

Selects 27 countries deterministically from a 3 by 3 grid of baseline country
size and baseline average radiance. Within each cell, the three observations
nearest the cell medians are retained. All four EOG annual bands are then
reduced over the same GAUL polygons at 1 km and 2 km; estimates use the main panel years (p0_config.py).

The 1 km run is a practical near-native check for a geographically varied
sample. It is not a new global headline estimate. Checkpointing occurs after
every country-scale call.

Outputs:
  data/p0_resolution_selection.csv
  data/p0_band_resolution.csv
"""

from __future__ import annotations

import os

import json
import signal
import socket
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaul_tiles import reduce_tiles, tile_geometries  # noqa: E402
from p0_config import VIIRS_START  # noqa: E402


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SELECTION_OUT = DATA / "p0_resolution_selection.csv"
OUT = DATA / "p0_band_resolution.csv"
RESULTS = BASE / "results_p0_resolution.json"
YEARS = range(2013, 2026)  # extracted years (cache); estimation uses the panel years below
PANEL_YEARS = range(VIIRS_START, 2026)
SCALES = (2000, 1000)
BANDS = ("average", "average_masked", "median", "median_masked")
PROJECT = os.environ.get('EE_PROJECT', '')
CALL_TIMEOUT = 900


class Timeout(Exception):
    pass


def alarm_handler(_signum, _frame):
    raise Timeout()


def select_countries() -> pd.DataFrame:
    annual = pd.read_csv(DATA / "country_ntl_annual.csv")
    panel = pd.read_csv(DATA / "global_panel.csv")
    eligible = set(panel.loc[panel["year"] == VIIRS_START, "gaul_code"])
    baseline = annual[
        (annual["year"] == VIIRS_START)
        & annual["gaul_code"].isin(eligible)
        & (annual["avg_mean"] > 0)
        & (annual["avg_count"] > 0)
    ].copy()
    baseline["log_size"] = np.log(baseline["avg_count"])
    baseline["log_brightness"] = np.log(baseline["avg_mean"])
    baseline["size_group"] = pd.qcut(
        baseline["log_size"], 3, labels=["small", "medium", "large"]
    )
    baseline["brightness_group"] = pd.qcut(
        baseline["log_brightness"], 3, labels=["dim", "middle", "bright"]
    )
    selected = []
    for (_size, _brightness), group in baseline.groupby(
        ["size_group", "brightness_group"], observed=True
    ):
        size_scale = group["log_size"].std() or 1.0
        light_scale = group["log_brightness"].std() or 1.0
        distance = np.square(
            (group["log_size"] - group["log_size"].median()) / size_scale
        ) + np.square(
            (group["log_brightness"] - group["log_brightness"].median())
            / light_scale
        )
        selected.append(group.loc[distance.nsmallest(3).index])
    selection = pd.concat(selected).sort_values(
        ["size_group", "brightness_group", "country"]
    )
    selection[
        [
            "country", "gaul_code", "avg_count", "avg_mean",
            "size_group", "brightness_group",
        ]
    ].to_csv(SELECTION_OUT, index=False)
    return selection


def image_stack() -> ee.Image:
    v21 = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21")
    v22 = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V22")
    images = []
    for year in YEARS:
        collection = v21 if year <= 2021 else v22
        image = (
            collection.filterDate(f"{year}-01-01", f"{year}-12-31")
            .first()
            .select(list(BANDS))
            .rename([f"{band}_{year}" for band in BANDS])
        )
        images.append(image)
    return ee.Image.cat(images)


def extract(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    completed = set()
    if OUT.exists():
        previous = pd.read_csv(OUT)
        rows = previous.to_dict("records")
        completed = set(zip(previous["gaul_code"], previous["scale"]))
    required = {
        (int(code), scale)
        for code in selection["gaul_code"]
        for scale in SCALES
    }
    selected = {int(code) for code in selection["gaul_code"]}

    def current(records) -> pd.DataFrame:
        """Rows for the current selection only; a changed selection leaves stale rows behind."""
        frame = pd.DataFrame(records)
        return frame[frame["gaul_code"].astype(int).isin(selected)].reset_index(drop=True)

    if required.issubset(completed):
        print("resolution extraction already complete; using bundled output", flush=True)
        data = current(rows)
        data.to_csv(OUT, index=False)
        return data

    socket.setdefaulttimeout(CALL_TIMEOUT)
    signal.signal(signal.SIGALRM, alarm_handler)
    ee.Initialize(project=PROJECT)
    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    stack = image_stack()
    band_names = [f"{band}_{year}" for year in YEARS for band in BANDS]

    for record in selection.itertuples():
        geometries = tile_geometries(countries, int(record.gaul_code), 2000)  # every GAUL tile
        for scale in SCALES:
            if (record.gaul_code, scale) in completed:
                continue
            print(f"{record.country} at {scale} m", flush=True)
            values = None
            start = time.time()
            for attempt in range(3):
                try:
                    signal.alarm(CALL_TIMEOUT)
                    tiles = reduce_tiles(
                        stack, geometries, band_names,
                        scale=scale,
                        crs="EPSG:4326",
                        bestEffort=False,
                        maxPixels=10_000_000_000,
                        tileScale=4 * (attempt + 1),
                    )
                    values = {name: tiles[f"{name}_mean"] for name in band_names}
                    signal.alarm(0)
                    break
                except Exception as error:
                    signal.alarm(0)
                    print(
                        f"  attempt {attempt + 1}: {type(error).__name__}",
                        flush=True,
                    )
                    if attempt < 2:
                        time.sleep(10)
            if values is None:
                print("  failed after three attempts", flush=True)
                continue
            for year in YEARS:
                row = {
                    "country": record.country,
                    "gaul_code": int(record.gaul_code),
                    "year": year,
                    "scale": scale,
                    "size_group": record.size_group,
                    "brightness_group": record.brightness_group,
                }
                for band in BANDS:
                    row[band] = values.get(f"{band}_{year}")
                rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  completed in {time.time() - start:.1f}s", flush=True)
    data = current(rows)
    data.to_csv(OUT, index=False)
    return data


def estimate(data: pd.DataFrame) -> dict:
    panel = pd.read_csv(DATA / "global_panel.csv")[["gaul_code", "year", "log_y"]]
    merged = data.merge(panel, on=["gaul_code", "year"], how="inner")
    results = {"selection": {}, "scales": {}, "scale_correlations": {}}
    for group, frame in data.groupby(["size_group", "brightness_group"]):
        results["selection"]["/".join(group)] = sorted(frame["country"].unique())

    for scale, frame in merged.groupby("scale"):
        scale_results = {}
        for band in BANDS:
            sample = frame[(frame[band] > 0) & frame["log_y"].notna()].copy()
            sample["log_l"] = np.log(sample[band])
            indexed = sample.set_index(["country", "year"]).sort_index()
            levels = PanelOLS(
                indexed["log_y"], indexed[["log_l"]],
                entity_effects=True, time_effects=True,
            ).fit(cov_type="clustered", cluster_entity=True)
            growth = sample.sort_values(["country", "year"]).copy()
            growth["dy"] = growth.groupby("country")["log_y"].diff()
            growth["dx"] = growth.groupby("country")["log_l"].diff()
            growth["gap"] = growth.groupby("country")["year"].diff()
            growth = growth[growth["gap"] == 1].dropna(subset=["dy", "dx"])
            growth_indexed = growth.set_index(["country", "year"]).sort_index()
            changes = PanelOLS(
                growth_indexed["dy"], growth_indexed[["dx"]], time_effects=True
            ).fit(cov_type="clustered", cluster_entity=True)
            scale_results[band] = {
                "levels_2wfe": float(levels.params["log_l"]),
                "levels_se": float(levels.std_errors["log_l"]),
                "growth_yearfe": float(changes.params["dx"]),
                "growth_se": float(changes.std_errors["dx"]),
                "n_levels": int(levels.nobs),
                "n_growth": int(changes.nobs),
            }
        results["scales"][str(int(scale))] = scale_results

    wide = data.pivot(
        index=["country", "gaul_code", "year"], columns="scale", values=list(BANDS)
    )
    for band in BANDS:
        pair = wide[band].where(wide[band] > 0).dropna()
        level_correlation = pair[1000].corr(pair[2000])
        changes = pair.groupby(level="country").apply(
            lambda group: np.log(group).diff()
        ).dropna()
        growth_correlation = changes[1000].corr(changes[2000])
        results["scale_correlations"][band] = {
            "level_1000_vs_2000": float(level_correlation),
            "growth_1000_vs_2000": float(growth_correlation),
        }
    return results


def main() -> None:
    selection = select_countries()
    print("selected countries:", ", ".join(selection["country"]), flush=True)
    data = extract(selection)
    data = data[data["year"].isin(PANEL_YEARS)].reset_index(drop=True)
    expected = len(selection) * len(SCALES) * len(PANEL_YEARS)
    if len(data) < expected:
        print(f"incomplete extraction: {len(data)}/{expected} rows", flush=True)
        return
    results = estimate(data)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["scales"], indent=2), flush=True)
    print(f"written: {OUT}, {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
