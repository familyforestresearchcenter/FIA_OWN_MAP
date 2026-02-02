import os
import gc
import math
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray
from tqdm import tqdm
from rasterio.windows import Window
import rasterio as rio
from geocube.api.core import make_geocube
from configs import *  # expects INPUT_DIR, etc.

TMP_DIR = "/dev/shm"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def polygons_to_raster(poly: gpd.GeoDataFrame, like_da: xr.DataArray) -> xr.DataArray:
    """Rasterize polygons aligned to `like_da` grid/extent; returns uint8 DataArray."""
    if poly.empty:
        return xr.zeros_like(like_da, dtype="uint8")
    ds = make_geocube(vector_data=poly, measurements=["Value"], like=like_da, fill=0)
    return ds["Value"].astype("uint8")


def build_owner_lut(owner_code, array_nlcd_dict, padus_join_code) -> np.ndarray:
    """256-length LUT (uint16) mapping current map codes -> new map codes for a given owner."""
    lut = np.arange(256, dtype=np.uint16)
    owner_map = padus_join_code.get(owner_code, {}) or padus_join_code.get(float(owner_code), {})
    for v in range(256):
        if v <= 85:
            lc = array_nlcd_dict.get(v, np.nan)
            if lc == lc and lc in owner_map:
                lut[v] = np.uint16(owner_map[lc])
    return lut


def apply_lut_da(ar_codes_da: xr.DataArray, lut: np.ndarray) -> xr.DataArray:
    """Chunk-friendly LUT mapping with xarray.apply_ufunc."""
    lut_da = xr.DataArray(lut, dims=("lut_index",))
    return xr.apply_ufunc(
        np.take, lut_da, ar_codes_da,
        input_core_dims=[["lut_index"], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[ar_codes_da.dtype],
    )


def coerce_to_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame has a CRS and matches `target_crs`."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4269")  # known source CRS
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # === 1) Discover state & load encoded raster
    gdb_file = next((f for f in os.listdir(TMP_DIR) if f.endswith(".gdb")), None)
    if gdb_file is None:
        raise FileNotFoundError("No .gdb file found in /dev/shm")
    state_name = gdb_file.split(".")[0]

    encoded_raster_path = os.path.join(TMP_DIR, "Own_Type_by_Landcover_Encoded.tif")
    src = rioxarray.open_rasterio(encoded_raster_path, masked=True).squeeze()
    src = src.rio.write_crs(4269)  # enforce EPSG:4269
    ar2_da = src.fillna(0).clip(min=0, max=255).astype("uint8")
    if not ar2_da.chunks:
        ar2_da = ar2_da.chunk({"y": 2048, "x": 2048})

    minx, miny, maxx, maxy = ar2_da.rio.bounds()

    # === 2) PADUS
    pad_gdf = gpd.read_file(os.path.join(INPUT_DIR, "PAD_US_Combined.shp"))
    pad_gdf = coerce_to_crs(pad_gdf, ar2_da.rio.crs)
    pad_gdf = pad_gdf.cx[minx:maxx, miny:maxy]
    pad_gdf["Value"] = pad_gdf["INC_Field"].astype(np.uint8)
    pad_da = polygons_to_raster(pad_gdf, ar2_da)
    if not pad_da.chunks:
        chunk_map = {dim: sizes[0] for dim, sizes in ar2_da.chunksizes.items()}
        pad_da = pad_da.chunk(chunk_map)

    # === 3) Dictionary
    dictionary = pd.read_excel(os.path.join(INPUT_DIR, "FIA_Ownership_Map_Data_Dictionary_Shareable.xlsx"))
    array_nlcd_dict = dict(zip(dictionary.Map_Value_Code, dictionary.Land_Cover_Code))
    padus_join_code = {}
    for own_code in dictionary.FIA_Ownership_Code.unique():
        padus_join_code[own_code] = {}
        for lc in dictionary.Land_Cover_Code.unique():
            match = dictionary[(dictionary.FIA_Ownership_Code == own_code) & (dictionary.Land_Cover_Code == lc)]
            if not match.empty:
                padus_join_code[own_code][lc] = match["Map_Value_Code"].values[0]

    # === 4) PADUS overlay
    base_cond = (ar2_da <= 85) & (pad_da > 0)
    ar_work = ar2_da.astype("uint16")
    for owner in padus_join_code.keys():
        if owner != owner:
            continue
        owner_int = int(owner) if (
            isinstance(owner, (int, np.integer)) or (isinstance(owner, float) and float(owner).is_integer())
        ) else owner
        cond_owner = base_cond & (pad_da == owner_int)
        if bool(cond_owner.any()):
            lut = build_owner_lut(owner_int, array_nlcd_dict, padus_join_code)
            mapped_for_owner = apply_lut_da(ar_work, lut).astype("uint16")
            ar_work = xr.where(cond_owner, mapped_for_owner, ar_work)
    del pad_da, pad_gdf
    gc.collect()

    # === 5) Indigenous Lands
    IL_gdf = gpd.read_file(os.path.join(INPUT_DIR, "Indigenous_Lands_BIA_AIAN_National_LAR.shp"))
    IL_gdf = coerce_to_crs(IL_gdf, ar2_da.rio.crs)
    IL_gdf = IL_gdf.cx[minx:maxx, miny:maxy]
    IL_gdf["Value"] = np.uint8(44)
    if not IL_gdf.empty:
        il_da = polygons_to_raster(IL_gdf, ar2_da)
        if not il_da.chunks:
            chunk_map = {dim: sizes[0] for dim, sizes in ar2_da.chunksizes.items()}
            il_da = il_da.chunk(chunk_map)
        cond_il = (ar_work <= 85) & (il_da == 44)
        lut44 = build_owner_lut(44, array_nlcd_dict, padus_join_code)
        mapped_44 = apply_lut_da(ar_work, lut44).astype("uint16")
        ar_work = xr.where(cond_il, mapped_44, ar_work)
        ar_work = ar_work.clip(min=0, max=255)
        ar_work = xr.where(ar_work >= 255, 0, ar_work).astype("uint16")
        del il_da, IL_gdf
        gc.collect()

    # === 6) Finalize
    ar2_da = ar_work.astype("uint8").squeeze(drop=True)
    ar2_da = ar2_da.chunk({"y": 4096, "x": 4096})
    # FIX: explicitly set CRS before writing, in case it's missing
    if ar2_da.rio.crs is None:
        ar2_da = ar2_da.rio.write_crs("EPSG:4269", inplace=False)
    else:
        ar2_da = ar2_da.rio.write_crs(ar2_da.rio.crs, inplace=False)

    height, width = ar2_da.sizes["y"], ar2_da.sizes["x"]
    transform, crs = ar2_da.rio.transform(), ar2_da.rio.crs

    # === 7) Output
    output_path = os.path.join(TMP_DIR, f"{state_name}_Final_Encoded.tif")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 1024,
        "blockysize": 1024,
        "compress": "zstd",
        "ZSTD_LEVEL": 5,
        "BIGTIFF": "YES",
        "NUM_THREADS": "ALL_CPUS",
    }
    super_sz = 4096
    tiles_x = math.ceil(width / super_sz)
    tiles_y = math.ceil(height / super_sz)
    total_tiles = tiles_x * tiles_y

    env_kwargs = dict(GDAL_CACHEMAX=1024, VSI_CACHE="TRUE", VSI_CACHE_SIZE=100_000_000)
    with rio.Env(**env_kwargs):
        with rio.open(output_path, "w", **profile) as dst, tqdm(total=total_tiles, unit="supertile") as pbar:
            for ty in range(tiles_y):
                y0, y1 = ty * super_sz, min((ty + 1) * super_sz, height)
                for tx in range(tiles_x):
                    x0, x1 = tx * super_sz, min((tx + 1) * super_sz, width)
                    block = ar2_da.isel(y=slice(y0, y1), x=slice(x0, x1)).values
                    if block.ndim == 3:
                        block = block[0]
                    elif block.ndim == 4:
                        block = np.squeeze(block)
                    if block.ndim != 2:
                        raise ValueError(f"Expected 2D tile, got shape {block.shape}")
                    block = np.ascontiguousarray(block, dtype=np.uint8)
                    dst.write(block, 1, window=Window(x0, y0, x1 - x0, y1 - y0))
                    pbar.update(1)
                    del block
                    gc.collect()

    print(f"\n✅ Final raster written to: {output_path}")
