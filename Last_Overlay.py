import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
# import numpy_indexed as npi
# from osgeo import gdal
import xarray
from geocube.api.core import make_geocube
from tqdm import tqdm
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rioxarray
from configs import *
from rasterio.crs import CRS
import psutil
import gc

def polygonsToRaster(poly, source):
    state_parcel_raster = make_geocube(
        vector_data = poly,
        measurements=["Value"],
        resolution=source.rio.resolution(),
        fill = 0
        )
        

    return state_parcel_raster
        
if __name__ == '__main__':

    for i in os.listdir(DATA_DIR):
        if len(i) == 2:
            State_name = i

    ras = f'{DATA_DIR}{State_name}/{State_name}_Own_Type_by_Landcover_Encoded.tif'

    pad_gdf = gpd.read_file(INPUT_DIR + "PAD_US_Combined.shp")

    src = rioxarray.open_rasterio(ras, masked=True).squeeze()

    src = src.rio.write_crs(4269)

    src = src.rio.reproject(pad_gdf.crs)

    pad_gdf = pad_gdf.cx[src.rio.bounds()[0]:src.rio.bounds()[2],src.rio.bounds()[1]:src.rio.bounds()[3]]

    pad_gdf['Value'] = pad_gdf['INC_Field'].astype(np.uint8)

    gc.collect()

    print('Starting Rasterization')

    raster2 = polygonsToRaster(pad_gdf, src)

    print("Pad Raster Created")

    IL_gdf = gpd.read_file(INPUT_DIR + 'Indigenous_Lands_BIA_AIAN_National_LAR.shp')

    IL_gdf = IL_gdf.to_crs(src.rio.crs)

    IL_gdf = IL_gdf.cx[src.rio.bounds()[0]:src.rio.bounds()[2],src.rio.bounds()[1]:src.rio.bounds()[3]]

    IL_gdf['Value'] = 44

    raster1 = src

    gc.collect()

    raster2 = raster2.Value

    print("Starting to align the rasters")

    raster2 = raster2.rio.reproject_match(raster1)

    # Resample raster2 to match the resolution of raster1
    raster2 = raster2.rio.reproject(raster1.rio.crs, resolution=raster1.rio.resolution())

    print('Pad Aligned')

    bbox = [raster1.rio.bounds()[0], raster1.rio.bounds()[1], raster1.rio.bounds()[2], raster1.rio.bounds()[3]]

    # Clip both rasters to the same bounding box
    clipped_raster1 = raster1.rio.clip_box(*bbox)
    clipped_raster2 = raster2.rio.clip_box(*bbox)

    print('SRC and PAD Rasters Clipped')

    ar2 = clipped_raster1.to_numpy()
    pad = clipped_raster2.to_numpy()


    # print(np.unique(array), np.unique(pad), np.unique(il))

    dictionary = pd.read_excel(INPUT_DIR + 'FIA_Ownership_Map_Data_Dictionary_Shareable.xlsx')

    array_nlcd_dict = dict(zip(dictionary.Map_Value_Code, dictionary.Land_Cover_Code))

    padus_join_code = {}

    for i in dictionary.FIA_Ownership_Code.unique():
        padus_join_code[i] = {}
        for l in dictionary.Land_Cover_Code.unique():
            if len(dictionary.loc[(dictionary.FIA_Ownership_Code == i) & (dictionary.Land_Cover_Code == l)]) >= 1:
                padus_join_code[i][l] = dictionary.loc[(dictionary.FIA_Ownership_Code == i) & (dictionary.Land_Cover_Code == l)]['Map_Value_Code'].values[0]


    print('PAD Recoded')

    if pad.shape != ar2.shape:
        pad = pad[:ar2.shape[0], :ar2.shape[1]]

    print('Pad Reshaped')

    aa = ar2 <= 85

    gc.collect()

    ab = pad > 0

    logical_array = np.logical_and(aa, ab)

    mask = np.where(logical_array == True)

    del aa, ab, logical_array
    gc.collect()

    for i in tqdm(range(len(mask[0]))):
        lc = array_nlcd_dict[ar2[mask[0][i]][mask[1][i]]]
        own = pad[mask[0][i]][mask[1][i]]
        ar2[mask[0][i]][mask[1][i]] =  padus_join_code[own][lc]

    del mask, pad, raster2, clipped_raster2
    gc.collect()

    print('PAD Overlay COmplete')

    if len(IL_gdf) > 0:

        raster3 = polygonsToRaster(IL_gdf, src)

        print('IL Rasterized')

        raster3 = raster3.Value

    # Reproject raster2 to match the CRS of raster1
        raster3 = raster3.rio.reproject_match(raster1)

        print('IL Reprojected')

    # Resample raster2 to match the resolution of raster1
        raster3 = raster3.rio.reproject(raster1.rio.crs, resolution=raster1.rio.resolution())

        clipped_raster3 = raster3.rio.clip_box(*bbox)

        del raster3

        print('IL Clipped')

        il = clipped_raster3.to_numpy()

        del clipped_raster3

        gc.collect()

        if il.shape != ar2.shape:
            il = il[:ar2.shape[0], :ar2.shape[1]]

            aaa = ar2 <= 85

            abb = il == 44

            logical_array2 = np.logical_and(aaa, abb)

            mask2 = np.where(logical_array2 == True)

            del aaa, abb, logical_array2

            gc.collect()

            for i in tqdm(range(len(mask2[0]))):
                lc = array_nlcd_dict[ar2[mask2[0][i]][mask2[1][i]]]
                own = 44
                ar2[mask2[0][i]][mask2[1][i]] =  padus_join_code[own][lc]


            ar2 = np.where(ar2 > 255, 0, ar2)

            del mask2, il

    
    ar2 = ar2.astype(np.uint8)

    gc.collect()

    os.remove(ras)

    xarray.DataArray(ar2, coords={'latitude': src['y'].values, 'longitude': src['x'].values},
        dims=['latitude', 'longitude']).rio.write_crs(pad_gdf.crs, inplace=True).rio.to_raster(f'{DATA_DIR}{State_name}/{State_name}_Final_Encoded.tif', dtype=np.uint8, tiled=True, windowed=True, compress='zstd')
    