import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import gc
import xarray
import rioxarray
from shapely.geometry import Polygon, mapping
from geopandas import GeoSeries
from rasterio.mask import mask
from geocube.api.core import make_geocube
from configs import *

def polygonsToRaster(poly):
    poly = poly.to_crs('EPSG:4326')
    return make_geocube(
        vector_data=poly,
        measurements=["Value"],
        resolution=(-9.96023190602395e-05, 9.96023190602395e-05),
        fill=-99999
    )

def fillFlag(x):
    return 0 if x['IL_Flag'] != 1 else x['IL_Flag']

def getXY(pt):
    return (pt.x, pt.y)

if __name__ == '__main__':
    TMP_DIR = '/dev/shm'


    for file in os.listdir(TMP_DIR):
        if ".gdb" in file:
            gdb_file = file
            State_name = gdb_file.split('.')[0]
            print("State:", State_name)

    gdb_path = os.path.join(TMP_DIR, gdb_file)

    try:
        parcels = gpd.read_file(gdb_path, driver='FileGDB', layer='parcels')
    except:
        parcels = gpd.read_file(gdb_path, driver='FileGDB', layer='Parcels')

    print('Step1: Points and parcels loaded')

    parcels['State_name'] = State_name
    parcels['Value'] = range(len(parcels))
    new_parcels = polygonsToRaster(parcels)

    # Save the id_values to a json
    raster_path = os.path.join(TMP_DIR, f"{State_name}_Parcel_IDs.json")
    # update the column name value to join_index
    parcels[['Value', 'geometry']].to_file(raster_path, driver='GeoJSON')

    print('Step2: Polygon to raster complete')

    NLCD = rasterio.open(INPUT_DIR + "NLCD_10m_NAD83.tif")
    bbox = [new_parcels.x.values.min(), new_parcels.y.values.min(), new_parcels.x.values.max(), new_parcels.y.values.max()]
    geometry = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])
    feature = [mapping(geometry)]

    id_array = new_parcels.to_array().values.astype('int_')
    clipped_NLCD, _ = mask(NLCD, feature, crop=True)
    clipped_NLCD = clipped_NLCD[0].astype('int8')
    clipped_NLCD = clipped_NLCD[0:id_array[0].shape[0]]

    del NLCD, new_parcels
    gc.collect()

    print('Step3: Clip NLCD complete')

    idx = np.where(id_array[0].flatten() == -99999)
    id_array = np.delete(id_array[0].flatten(), idx)
    reclass_NLCD = np.delete(clipped_NLCD.flatten(), idx)

    df = pd.DataFrame({'Value': id_array.flatten(), 'NLCD': reclass_NLCD.flatten()})
    del id_array, reclass_NLCD, idx
    gc.collect()

    print('Step4: Align ID and NLCD complete')

    cdf = pd.DataFrame(columns=['Value', 'NLCD', 'counts'])
    s, e = 0, 10000000
    while s < len(df):
        if e > len(df):
            e = len(df)
        temp = df.iloc[s:e]
        temp = temp.value_counts().rename_axis(['Value', 'NLCD']).reset_index(name='counts').astype('int32')
        cdf = pd.concat([cdf, temp])
        s += 10000000
        e += 10000000
        del temp
        gc.collect()
    del df
    gc.collect()

    print('Step5: Count aggregation complete')

    sums = cdf.groupby(['Value'])['counts'].sum()
    df = pd.DataFrame(cdf.groupby(['Value', 'NLCD'])['counts'].sum())
    del cdf
    gc.collect()

    df['Value'] = df.index.get_level_values('Value')
    df['NLCD'] = df.index.get_level_values('NLCD')
    df = df.droplevel('NLCD')
    df = df.pivot(index='Value', columns='NLCD', values='counts').fillna(0)
    df.columns = df.columns.to_series().map(reclass_dict)
    df['Total_Count'] = sums
    df = df.iloc[:, 1:].div(df.Total_Count, axis=0).drop('Total_Count', axis=1)

    print('Step6: Land cover proportions complete')

    counties = gpd.read_file(INPUT_DIR + 'tl_2020_us_county/tl_2020_us_county.shp')
    counties = counties.to_crs(4269)
    parcels = parcels.to_crs(5070)
    centroids = GeoSeries(parcels['geometry']).centroid
    x, y = [list(t) for t in zip(*map(getXY, centroids))]
    parcels['Centroid'] = centroids
    parcels['Centroid_X'] = x
    parcels['Centroid_Y'] = y
    parcels["PARCEL_AREA"] = parcels['geometry'].area / 10**6
    print("Step7: Parcel geometry calculated")

    counties = counties[['COUNTYFP', 'NAME', 'geometry']]
    parcels = parcels.set_geometry('Centroid').to_crs(4269)
    gdf = gpd.sjoin(parcels, counties, how="left", predicate='within')
    gdf = gdf.set_geometry('geometry').to_crs(4269).drop(['index_right'], axis=1)
    gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME', 'COUNTYFP': 'COUNTY_FIPS'})

    for idx in gdf[gdf.COUNTY_NAME.isnull()].index:
        temp = gdf.loc[[idx]]
        temp_join = gpd.sjoin(temp, counties, how="left", predicate='intersects')
        gdf.at[idx, 'COUNTY_NAME'] = temp_join.NAME.values[0]
        gdf.at[idx, 'COUNTY_FIPS'] = temp_join.COUNTYFP.values[0]

    polygon = gdf
    print('Step8: County info joined')

    ILs = gpd.read_file(INPUT_DIR + 'Indigenous_Lands_BIA_AIAN_National_LAR.shp')
    ILs = ILs[['Name', 'geometry']]
    ILs['IL_Flag'] = 1
    ILs = ILs.to_crs(4269)

    polygon = polygon.set_geometry('Centroid')
    polygon = gpd.sjoin(polygon, ILs, how="left", predicate='within')
    polygon['IL_Flag'] = polygon.apply(fillFlag, axis=1)
    polygon = polygon.drop_duplicates(subset=['PRCLDMPID'], keep='first')

    print('Step9: Indigenous Lands flagged')

    pad = gpd.read_file(INPUT_DIR + "PAD_US_Combined.shp").to_crs(4269)
    pad = pd.concat([
        pad.loc[pad.INC_Field == 25].assign(GOV_Flag=2),
        pad.loc[pad.INC_Field == 31].assign(GOV_Flag=3),
        pad.loc[pad.INC_Field == 32].assign(GOV_Flag=4)
    ])
    pad = pad[['GOV_Flag', "geometry"]]

    def gov_fill(x):
        return x.IL_Flag if x.IL_Flag == 1 else x.GOV_Flag

    polygon = gpd.sjoin(polygon.drop(columns=['index_right']), pad, how="left", predicate='within')
    polygon['IL_Flag'] = polygon.apply(gov_fill, axis=1)

    print('Step10: PAD overlay joined')

    points = gpd.read_file(gdb_path, driver='FileGDB', layer='Propertypoints')
    points = points[['OWN1', 'OWN2', 'MCAREOFNAM', 'MHSNUMB', 'MPREDIR', 'MSTNAME', 'MMODE', 'PRCLDMPID']]
    points = points.drop_duplicates(['PRCLDMPID'], keep='first')

    print('Step11: Ownership points loaded')

    state = polygon.merge(pd.DataFrame(points), on='PRCLDMPID', how='left')
    state = pd.DataFrame(state.drop(columns='geometry'))

    del counties, ILs, centroids, gdf, parcels, points, polygon
    gc.collect()

    full_state = pd.merge(state, df, on='Value', how='left')
    full_state = full_state.rename(columns={
        0:'Unclassified', 1:'Open Water', 2:'Developed, Open Space',
        3:'Developed, Low Intensity', 4:'Developed, Medium Intensity',
        5:'Developed, High Intensity', 6:'Barren Land', 7:'Deciduous Forest',
        8:'Evergreen Forest', 9:'Mixed Forest', 11:'Shrub/Scrub',
        12:'Herbaceuous', 13:'Hay/Pasture', 14:'Cultivated Crops',
        15:'Woody Wetlands', 16:'Emergent Herbaceuous Wetlands',
        -99:'Unclassified', 19:'Perennial Ice/Snow'
    })

    full_state.to_csv('/dev/shm/temp.csv')
    print('✅ Preprocessing complete — Output saved to /dev/shm/temp.csv')
