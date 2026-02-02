import os
import gc
import pickle
import shutil
import rasterio
import xarray
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
from geocube.api.core import make_geocube
import numpy_indexed as npi
from configs import *

TMP_DIR = '/dev/shm'


def polygonsToRaster(poly):
    return make_geocube(
        vector_data=poly,
        measurements=["OWNCD"],
        resolution=(-9.96023190602395e-05, 9.96023190602395e-05),
        fill=-1
    )


if __name__ == '__main__':
    # === Find GDB File in /tmp/
    for file in os.listdir(TMP_DIR):
        if file.endswith('.gdb'):
            gdb_file = file
            State_name = gdb_file.split('.')[0]
            break
    else:
        raise FileNotFoundError("No .gdb file found in /tmp/")

    gdb_path = os.path.join(TMP_DIR, gdb_file)
    parcels = gpd.read_file(gdb_path, driver='FileGDB', layer='parcels')

    # === Read Reduced Ownership Table
    full_table_path = os.path.join(TMP_DIR, 'Full_Data_Table.csv')
    df = pd.read_csv(full_table_path)
    parcels["OWNCD"] = parcels.PRCLDMPID.map(df.set_index('PRCLDMPID').OWNCD.to_dict()).fillna(-99).astype('int8')

    # Optional: remove the GDB after use
    # shutil.rmtree(gdb_path, ignore_errors=True)
    gc.collect()

    # === Rasterize Parcels
    new_parcels = polygonsToRaster(parcels)

    # === Load and Clip NLCD Raster
    NLCD = rasterio.open(INPUT_DIR + "NLCD_10m_NAD83.tif")
    bbox = [new_parcels.x.values.min(), new_parcels.y.values.min(), new_parcels.x.values.max(), new_parcels.y.values.max()]
    geometry = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])
    feature = [mapping(geometry)]

    OWNCD_array = new_parcels.to_array().values.astype('int_')[0]
    clipped_NLCD, _ = mask(NLCD, feature, crop=True)
    clipped_NLCD = clipped_NLCD[0].astype('int8')

    del parcels, df
    gc.collect()

    coded_ar = OWNCD_array * 100 + clipped_NLCD
    del OWNCD_array, clipped_NLCD
    gc.collect()

    # === Load Reclassification Dictionary
    with open(INPUT_DIR + "New_Raster_Reclass.pickle", 'rb') as handle:
        new_recode = pickle.load(handle)

    # === Recode Raster
    new_array = np.zeros(coded_ar.shape, dtype=np.uint8)
    for r in tqdm(range(len(coded_ar))):
        new_array[r] = npi.remap(
            coded_ar[r].flatten(),
            list(new_recode.keys()),
            list(new_recode.values())
        ).reshape(coded_ar[r].shape)
    new_array[new_array == 32767] = 0

    # === Write Final Raster to /tmp/
    output_path = os.path.join(TMP_DIR, 'Own_Type_by_Landcover_Encoded.tif')
    xarray.DataArray(
        new_array,
        coords={'latitude': new_parcels['y'].values, 'longitude': new_parcels['x'].values},
        dims=['latitude', 'longitude']
    ).rio.to_raster(output_path, dtype=np.uint8, tiled=True, windowed=True, compress='zstd')

    print(f"\n✅ Map TIF created and written to: {output_path}")
