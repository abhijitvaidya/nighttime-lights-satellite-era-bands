"""Country reductions over every GAUL-2015 feature of a country.

FAO GAUL-2015 level 0 stores a few countries as several features (map tiles): Canada (10),
the United States (5), Australia (3) and the West Bank (2). Taking `.first()` for a country
code reduces only one tile. These helpers reduce each tile separately and combine the tile
means with weights equal to tile area in EPSG:4326 (square degrees), which is the weight each
tile carries in a whole-country mean over equal-angle pixels. Pixel counts are not used as
weights: with `bestEffort=True`, Earth Engine evaluates each region at the coarsest scale its
pixel limit requires, and the limit counts pixels across all bands of the call, so tiles of one
country can be evaluated at different scales and their counts are not comparable. For
single-feature countries the result is identical to the one-feature reduction.
"""
from __future__ import annotations

import ee

AREA_PROJ = "EPSG:4326"


def tile_geometries(countries: ee.FeatureCollection, code: int, simplify_m: float) -> list:
    """Simplified geometry of every GAUL feature with this ADM0_CODE (usually one)."""
    tiles = countries.filter(ee.Filter.eq("ADM0_CODE", code))
    ids = tiles.aggregate_array("system:index").getInfo()
    return [ee.Feature(tiles.filter(ee.Filter.eq("system:index", i)).first()).simplify(simplify_m).geometry()
            for i in ids]


def reduce_tile(image: ee.Image, geom, bands: list[str], **kwargs) -> dict:
    """Mean and count of each band over one tile, with the tile's area in square degrees.

    `kwargs` are passed to reduceRegion (scale, crs, bestEffort, maxPixels, tileScale).
    Returns {'area': value, band + '_mean': value, band + '_count': value}.
    """
    reducer = (ee.Reducer.mean().setOutputs(["mean"])
               .combine(ee.Reducer.count().setOutputs(["count"]), "", True))
    out = ee.Dictionary({"r": image.select(bands).reduceRegion(reducer=reducer, geometry=geom, **kwargs),
                         "area": geom.area(1, ee.Projection(AREA_PROJ))}).getInfo()
    res = {"area": out["area"]}
    for b in bands:
        res[f"{b}_mean"], res[f"{b}_count"] = out["r"].get(f"{b}_mean"), out["r"].get(f"{b}_count")
    return res


def combine(tiles: list[dict], bands: list[str]) -> dict:
    """Area-weighted mean and total evaluated pixel count of each band over a country's tiles."""
    out = {}
    for b in bands:
        ok = [t for t in tiles if t.get(f"{b}_mean") is not None and t.get(f"{b}_count")]
        w = sum(t["area"] or 0 for t in ok)
        if not ok:
            out[f"{b}_mean"] = None
        elif len(ok) == 1:  # one tile: its own mean, as the single-feature reduction gives
            out[f"{b}_mean"] = ok[0][f"{b}_mean"]
        elif w > 0:
            out[f"{b}_mean"] = sum(t[f"{b}_mean"] * (t["area"] or 0) for t in ok) / w
        else:  # degenerate areas (tiny islands): fall back to pixel-count weights
            n = sum(t[f"{b}_count"] for t in ok)
            out[f"{b}_mean"] = sum(t[f"{b}_mean"] * t[f"{b}_count"] for t in ok) / n
        out[f"{b}_count"] = sum(t[f"{b}_count"] for t in ok) if ok else None
    return out


def reduce_tiles(image: ee.Image, geoms: list, bands: list[str], **kwargs) -> dict:
    """Area-weighted mean and total count of each band over all tiles of a country.

    Returns {band + '_mean': value, band + '_count': value}, as the single-tile call does.
    """
    return combine([reduce_tile(image, g, bands, **kwargs) for g in geoms], bands)
