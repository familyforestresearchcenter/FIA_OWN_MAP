import pandas as pd
import geopandas as gpd
import os
import rasterio
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import gc
from shapely.geometry import box
import rasterio.mask
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from shapely.geometry import Polygon
from shapely.geometry import mapping
from rasterio.mask import mask
import xarray
from configs import * 
import shutil
from zipfile import ZipFile
import aspose.words as aw
import tarfile
import pickle
import numpy_indexed as npi


def polygonsToRaster(poly):
    polygons = poly
#     polygons = polygons.to_crs('EPSG:4269')
# #     polygons = polygons.to_crs('EPSG:4326')
    
    # Using GeoCube to rasterize the Vector
    state_parcel_raster = make_geocube(
        vector_data = polygons,
        measurements=["OWNCD"],
        resolution=(-9.96023190602395e-05, 9.96023190602395e-05),
        fill = -1
    )
    
    # Save raster census raster
    return state_parcel_raster


# def recode_map(coded_ar):
# # #Unmarked Public/Unknown
# #     map_codes = np.where(coded_ar == -9999900, 1, coded_ar)
# #     map_codes = np.where(((map_codes >= -9999859) & (map_codes <= -9999858)) | (map_codes == -9999857) | (map_codes == -9999818), 2, map_codes)
# #     map_codes = np.where(map_codes <= -9999805, 3, map_codes)

# #Unmarked Public/Unknown
#     map_codes = np.where(coded_ar == -100, 1, coded_ar)
#     map_codes = np.where(((map_codes >= -59) & (map_codes <= -58)) | (map_codes == -57) | (map_codes == -10), 2, map_codes)
#     map_codes = np.where(map_codes <= -5, 3, map_codes)

# #Null ownership types
#     map_codes = np.where(map_codes == -9900, 1, map_codes)
#     map_codes = np.where(((map_codes >= -9859) & (map_codes <= -9858)) | (map_codes == -9857) | (map_codes == -9890), 2, map_codes)
#     map_codes = np.where(map_codes <= -9805, 3, map_codes)

# #Public
#     map_codes = np.where(map_codes == 2500, 1, map_codes)
#     map_codes = np.where(((map_codes >= 2541) & (map_codes <= 2542)) | (map_codes == 2543) | (map_codes == 2590), 4, map_codes)
#     map_codes = np.where((map_codes <= 2595) & (map_codes >= 2499), 5, map_codes)

#     map_codes = np.where(map_codes == 3100, 1, map_codes)
#     map_codes = np.where(((map_codes >= 3141) & (map_codes <= 3142)) | (map_codes == 3143) | (map_codes == 3190), 4, map_codes)
#     map_codes = np.where((map_codes <= 3195) & (map_codes >= 3099), 5, map_codes)

#     map_codes = np.where(map_codes == 3200, 1, map_codes)
#     map_codes = np.where(((map_codes >= 3241) & (map_codes <= 3242)) | (map_codes == 3243) | (map_codes == 3290), 4, map_codes)
#     map_codes = np.where((map_codes <= 3295) & (map_codes >= 3199), 5, map_codes)

# #Corp
#     map_codes = np.where(map_codes == 4100, 1, map_codes)
#     map_codes = np.where(((map_codes >= 4141) & (map_codes <= 4142)) | (map_codes == 4143) | (map_codes == 4190), 6, map_codes)
#     map_codes = np.where((map_codes <= 4195) & (map_codes >= 4099), 7, map_codes)

# #Other Private
#     map_codes = np.where(map_codes == 4200, 1, map_codes)
#     map_codes = np.where(((map_codes >= 4241) & (map_codes <= 4242)) | (map_codes == 4243) | (map_codes == 4290), 8, map_codes)
#     map_codes = np.where((map_codes <= 4295) & (map_codes >= 4199), 9, map_codes)

#     map_codes = np.where(map_codes == 4300, 1, map_codes)
#     map_codes = np.where(((map_codes >= 4341) & (map_codes <= 4342)) | (map_codes == 4343) | (map_codes == 4390), 8, map_codes)
#     map_codes = np.where((map_codes <= 4395) & (map_codes >= 4299), 9, map_codes)

# #Indigenous
#     map_codes = np.where(map_codes == 4400, 1, map_codes)
#     map_codes = np.where(((map_codes >= 4441) & (map_codes <= 4442)) | (map_codes == 4443) | (map_codes == 4490), 10, map_codes)
#     map_codes = np.where((map_codes <= 4495) & (map_codes >= 4399), 11, map_codes)

# #Family
#     map_codes = np.where(map_codes == 4500, 1, map_codes)
#     map_codes = np.where(((map_codes >= 4541) & (map_codes <= 4542)) | (map_codes == 4543) | (map_codes == 4590), 12, map_codes)
#     map_codes = np.where((map_codes <= 4595) & (map_codes >= 4499), 13, map_codes)
    
#     return map_codes


if __name__ == '__main__':

    # os.chdir(r'D:\Documents\OwnershipMap\New_Script\Test_ENV\OUTPUTS\INTERMEDIATE')

    # state = "ME"
    
    for file in os.listdir(DATA_DIR):
            if ".gdb" in file:
                gdb_file = file
                State_name = gdb_file.split('.')[0]
                print(State_name)
                
                
    parcels = gpd.read_file(DATA_DIR + gdb_file, driver='FileGDB', layer='parcels')

    df = pd.read_csv(DATA_DIR + State_name + "//" + f'{State_name}_Reduced_Data_Table.csv') 
    
    # df["OWNCD"] = df["Own_Type"]
    
    parcels["OWNCD"] = parcels.PRCLDMPID.map(df.set_index('PRCLDMPID').OWNCD.to_dict()).fillna(-99).astype('int8')
    
    # import psutil
    # print(psutil.virtual_memory())
    
    shutil.rmtree(f'./data/{State_name}.gdb')
    gc.collect()
    
    new_parcels = polygonsToRaster(parcels)

    NLCD  = rasterio.open(INPUT_DIR + "NLCD_10m_NAD83.tif")
    bbox = [new_parcels.x.values.min(), new_parcels.y.values.min(), new_parcels.x.values.max(), new_parcels.y.values.max()]
    geometry = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])
    feature = [mapping(geometry)] # Required conversion

    OWNCD_array = new_parcels.to_array().values.astype('int_')[0]

    clipped_NLCD, out_transform = mask(NLCD, feature, crop=True)

    clipped_NLCD = clipped_NLCD[0].astype('int8')

    del parcels
    gc.collect()

    OWNCD_array = OWNCD_array*100

    coded_ar = OWNCD_array + clipped_NLCD

    del OWNCD_array, clipped_NLCD, df
    gc.collect()

    with open(INPUT_DIR + "New_Raster_Reclass.pickle", 'rb') as handle:
        new_recode = pickle.load(handle)

    new_array = np.zeros(coded_ar.shape, dtype = np.uint8)
    for r in tqdm(range(len(coded_ar))):
        new_array[r] = npi.remap(coded_ar[r].flatten(), list(new_recode.keys()), list(new_recode.values())).reshape(coded_ar[r].shape)
    new_array[new_array == 32767] = 0

    # # #Full_Code_tiff
    # xarray.DataArray(coded_ar, coords={'latitude': new_parcels['y'].values, 'longitude': new_parcels['x'].values},
    #             dims=['latitude', 'longitude']).rio.to_raster(f'{DATA_DIR}{State_name}/{State_name}_Own_Type_by_Landcover_Full.tif', dtype=np.int16, tiled=True, windowed=True, compress='zstd')
    
    xarray.DataArray(new_array, coords={'latitude': new_parcels['y'].values, 'longitude': new_parcels['x'].values},
                dims=['latitude', 'longitude']).rio.to_raster(f'{DATA_DIR}{State_name}/{State_name}_Own_Type_by_Landcover_Encoded.tif', dtype=np.uint8, tiled=True, windowed=True, compress='zstd')
    

    print("Map Tif Created")