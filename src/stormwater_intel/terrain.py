"""Community terrain features from the public Copernicus GLO-30 DEM (30 m surface model).

Tiles are cloud-optimised GeoTIFFs on AWS open data; only the blocks covering Calgary are
streamed. It is a surface model (buildings and trees included), so treat the features as a
coarse stop-gap for the City's 2 m LiDAR DEM (see docs/MANUAL_DATA.md).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COP30 = "https://copernicus-dem-30m.s3.amazonaws.com"
TERRAIN_FEATURES = ["elev_mean_m", "elev_relief_m", "slope_mean_deg", "tpi3km_mean_m", "low_frac"]
LOW_CELL_M = -5.0     # a cell 5 m below its 3 km neighbourhood mean counts as low-lying
TPI_WINDOW_M = 3000.0


def tile_url(lat0: int, lon0_abs: int) -> str:
    name = f"Copernicus_DSM_COG_10_N{lat0:02d}_00_W{lon0_abs:03d}_00_DEM"
    return f"/vsicurl/{COP30}/{name}/{name}.tif"


def download_dem(bounds: tuple[float, float, float, float], out: Path) -> Path:
    """Mosaic the GLO-30 tiles covering `bounds` (west, south, east, north in degrees) into one GeoTIFF."""
    import rasterio
    from rasterio.merge import merge

    west, south, east, north = bounds
    urls = [tile_url(lat, abs(lon)) for lat in range(int(np.floor(south)), int(np.floor(north)) + 1)
            for lon in range(int(np.floor(west)), int(np.floor(east)) + 1)]
    srcs = [rasterio.open(u) for u in urls]
    try:
        mosaic, transform = merge(srcs, bounds=bounds)
        nodata = srcs[0].nodata
    finally:
        for s in srcs:
            s.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata,
                       compress="deflate") as dst:
        dst.write(mosaic[0].astype("float32"), 1)
    log.info("DEM mosaic %s from %d tiles -> %s", mosaic.shape[1:], len(urls), out.name)
    return out


def terrain_features(dem_path: Path, communities) -> pd.DataFrame:
    """Per-community terrain summary (index = comm_code). `communities` is a GeoDataFrame in EPSG:4326."""
    import rasterio
    from rasterio import features
    from scipy import ndimage

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        transform, nodata, bounds = src.transform, src.nodata, src.bounds
    if nodata is not None:
        dem[dem == nodata] = np.nan
    lat_mid = (bounds.top + bounds.bottom) / 2
    dx = abs(transform.a) * 111_320.0 * np.cos(np.radians(lat_mid))   # metres per pixel, east-west
    dy = abs(transform.e) * 110_574.0                                  # metres per pixel, north-south
    gy, gx = np.gradient(dem, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    window = int(round(TPI_WINDOW_M / dx)) | 1
    rel = dem - ndimage.uniform_filter(dem, size=window, mode="nearest")

    geoms = communities.reset_index(drop=True).geometry
    labels = features.rasterize(((g, i) for i, g in enumerate(geoms)), out_shape=dem.shape, transform=transform,
                                fill=-1, dtype="int32", all_touched=True)
    frame = pd.DataFrame({"label": labels.ravel(), "elev": dem.ravel(), "slope": slope.ravel(), "rel": rel.ravel()})
    frame = frame[frame["label"] >= 0]
    grouped = frame.groupby("label")
    out = pd.DataFrame({
        "elev_mean_m": grouped["elev"].mean(),
        "elev_relief_m": grouped["elev"].quantile(0.95) - grouped["elev"].quantile(0.05),
        "slope_mean_deg": grouped["slope"].mean(),
        "tpi3km_mean_m": grouped["rel"].mean(),
        "low_frac": grouped["rel"].apply(lambda s: float((s < LOW_CELL_M).mean())),
    })
    out.index = communities.reset_index(drop=True)["comm_code"].iloc[out.index].to_numpy()
    return out[TERRAIN_FEATURES]


def load_terrain(dem_path: Path, communities) -> pd.DataFrame:
    """Terrain features, or all-NaN columns when the DEM (or rasterio) is unavailable."""
    codes = list(communities["comm_code"])
    try:
        if not Path(dem_path).exists():
            raise FileNotFoundError(dem_path)
        return terrain_features(dem_path, communities).reindex(codes)
    except Exception as exc:  # noqa: BLE001
        log.warning("terrain features unavailable (%s); continuing without them", exc)
        return pd.DataFrame(np.nan, index=codes, columns=TERRAIN_FEATURES)
