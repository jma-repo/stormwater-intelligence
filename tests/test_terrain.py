import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from stormwater_intel import terrain as T

DX, DY = 0.000417, 0.000278          # degrees, roughly the Copernicus 30 m grid at 51 N
WEST, NORTH = -114.10, 51.10


def _write_dem(path, elev: np.ndarray):
    with rasterio.open(path, "w", driver="GTiff", height=elev.shape[0], width=elev.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(WEST, NORTH, DX, DY),
                       nodata=-9999.0) as dst:
        dst.write(elev.astype("float32"), 1)


def _block(col0, col1, row0=20, row1=100):
    return box(WEST + col0 * DX, NORTH - row1 * DY, WEST + col1 * DX, NORTH - row0 * DY)


def test_terrain_features_on_a_tilted_plane(tmp_path):
    elev = 1000 + np.tile(np.arange(120, dtype=float), (120, 1))       # +1 m per pixel towards the east
    path = tmp_path / "dem.tif"
    _write_dem(path, elev)
    comm = gpd.GeoDataFrame({"comm_code": ["LFT", "RGT"]}, geometry=[_block(5, 50), _block(70, 115)], crs="EPSG:4326")
    t = T.terrain_features(path, comm)
    assert list(t.columns) == T.TERRAIN_FEATURES
    assert t.loc["RGT", "elev_mean_m"] > t.loc["LFT", "elev_mean_m"] + 40
    metres_per_pixel = DX * 111_320 * np.cos(np.radians(51.08))
    assert t.loc["LFT", "slope_mean_deg"] == pytest.approx(np.degrees(np.arctan(1 / metres_per_pixel)), rel=0.05)
    assert t.loc["LFT", "tpi3km_mean_m"] < 0 < t.loc["RGT", "tpi3km_mean_m"]   # low side vs high side of the plane


def test_load_terrain_degrades_to_nan_columns_when_the_dem_is_missing(tmp_path):
    comm = gpd.GeoDataFrame({"comm_code": ["AAA", "BBB"]}, geometry=[_block(5, 50), _block(70, 115)], crs="EPSG:4326")
    t = T.load_terrain(tmp_path / "nope.tif", comm)
    assert list(t.index) == ["AAA", "BBB"] and t.isna().all().all()
    assert isinstance(t, pd.DataFrame) and list(t.columns) == T.TERRAIN_FEATURES


def test_tile_url_naming_matches_copernicus_convention():
    url = T.tile_url(51, 114)
    assert url.endswith("Copernicus_DSM_COG_10_N51_00_W114_00_DEM/Copernicus_DSM_COG_10_N51_00_W114_00_DEM.tif")
    assert url.startswith("/vsicurl/https://")
