import os
import gc
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray
from tqdm import tqdm
import rioxarray
from configs import *
from rasterio.crs import CRS
from geocube.api.core import make_geocube

TMP_DIR = '/dev/shm'


def polygonsToRaster(poly, source):
    return make_geocube(
        vector_data=poly,
        measurements=["Value"],
        resolution=source.rio.resolution(),
        fill=0
    )


if __name__ == '__main__':
    # === Get state_name from .gdb file in /tmp/
    gdb_file = next((f for f in os.listdir(TMP_DIR) if f.endswith('.gdb')), None)
    if gdb_file is None:
        raise FileNotFoundError("No .gdb file found in /tmp/")
    state_name = gdb_file.split('.')[0]

    # === Load encoded raster from /tmp/
    encoded_raster_path = os.path.join(TMP_DIR, 'Own_Type_by_Landcover_Encoded.tif')
    src = rioxarray.open_rasterio(encoded_raster_path, masked=True).squeeze()
    src = src.rio.write_crs(4269)

    # === Load PADUS
    pad_gdf = gpd.read_file(os.path.join(INPUT_DIR, "PAD_US_Combined.shp"))
    pad_gdf = pad_gdf.to_crs(src.rio.crs)
    minx, miny, maxx, maxy = src.rio.bounds()
    pad_gdf = pad_gdf.cx[minx:maxx, miny:maxy]
    pad_gdf['Value'] = pad_gdf['INC_Field'].astype(np.uint8)

    # === Rasterize PADUS
    raster2 = polygonsToRaster(pad_gdf, src).Value
    raster2 = raster2.rio.reproject_match(src)

    bbox = src.rio.bounds()
    clipped_raster1 = src.rio.clip_box(*bbox)
    clipped_raster2 = raster2.rio.clip_box(*bbox)

    ar2 = clipped_raster1.to_numpy()
    pad = clipped_raster2.to_numpy()

    # === Load map value dictionary
    dictionary = pd.read_excel(os.path.join(INPUT_DIR, 'FIA_Ownership_Map_Data_Dictionary_Shareable.xlsx'))
    array_nlcd_dict = dict(zip(dictionary.Map_Value_Code, dictionary.Land_Cover_Code))

    padus_join_code = {}
    for own_code in dictionary.FIA_Ownership_Code.unique():
        padus_join_code[own_code] = {}
        for lc in dictionary.Land_Cover_Code.unique():
            match = dictionary[(dictionary.FIA_Ownership_Code == own_code) & (dictionary.Land_Cover_Code == lc)]
            if not match.empty:
                padus_join_code[own_code][lc] = match['Map_Value_Code'].values[0]

    if pad.shape != ar2.shape:
        pad = pad[:ar2.shape[0], :ar2.shape[1]]

    # === Overlay PADUS
    mask = np.where((ar2 <= 85) & (pad > 0))
    for i in tqdm(range(len(mask[0]))):
        row, col = mask[0][i], mask[1][i]
        lc = array_nlcd_dict.get(ar2[row, col])
        own = pad[row, col]
        if lc in padus_join_code.get(own, {}):
            ar2[row, col] = padus_join_code[own][lc]

    del mask, pad, raster2, clipped_raster2
    gc.collect()

    # === Indigenous Lands overlay
    IL_gdf = gpd.read_file(os.path.join(INPUT_DIR, 'Indigenous_Lands_BIA_AIAN_National_LAR.shp')).to_crs(src.rio.crs)
    IL_gdf = IL_gdf.cx[minx:maxx, miny:maxy]
    IL_gdf['Value'] = 44

    if not IL_gdf.empty:
        raster3 = polygonsToRaster(IL_gdf, src).Value
        raster3 = raster3.rio.reproject_match(src)
        clipped_raster3 = raster3.rio.clip_box(*bbox)
        il = clipped_raster3.to_numpy()

        if il.shape != ar2.shape:
            il = il[:ar2.shape[0], :ar2.shape[1]]

        mask2 = np.where((ar2 <= 85) & (il == 44))
        for i in tqdm(range(len(mask2[0]))):
            row, col = mask2[0][i], mask2[1][i]
            lc = array_nlcd_dict.get(ar2[row, col])
            if lc in padus_join_code.get(44, {}):
                ar2[row, col] = padus_join_code[44][lc]

        ar2 = np.nan_to_num(ar2, nan=0)
        ar2 = np.clip(ar2, 0, 255)
        ar2 = np.where(ar2 >= 255, 0, ar2)

        del mask2, il, clipped_raster3

    # === Final output
    ar2 = ar2.astype(np.uint8)
    gc.collect()

    output_path = os.path.join(TMP_DIR, f'{state_name}_Final_Encoded.tif')
    xarray.DataArray(
        ar2,
        coords={'latitude': src['y'].values, 'longitude': src['x'].values},
        dims=['latitude', 'longitude']
    ).rio.write_crs(pad_gdf.crs, inplace=True).rio.to_raster(output_path, dtype=np.uint8, tiled=True, windowed=True, compress='zstd')

    print(f"\n✅ Final raster written to: {output_path}")
