from osgeo import ogr
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
from numpy import int16
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from functools import partial
from rasterio.enums import MergeAlg
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import gc
from geopandas import GeoSeries
from configs import *

def polygonsToRaster(poly):
    polygons = poly
    polygons = polygons.to_crs('EPSG:4326')
    
    # Using GeoCube to rasterize the Vector
    state_parcel_raster = make_geocube(
        vector_data = polygons,
        measurements=["Value"],
        resolution=(-9.96023190602395e-05, 9.96023190602395e-05),
        fill = -99999
    )
    
    # Save raster census raster
    return state_parcel_raster

reclass_dict = {
#     Unclassified
        0: -99,
#     'Open Water'
        11:  1,
#     'Developed, Open Space'
        21: 2,
#     'Developed, Low Intensity'
        22: 3,
#     'Developed, Medium Intensity'
        23: 4,
#     'Developed, High Intensity'
        24: 5,
#     'Barren Land'
        31: 6,
#     'Deciduous Forest'
        41: 7,
#     'Evergreen Forest'
        42: 8,
#     'Mixed Forest'
        43: 9,
#     'Shrub/Scrub'
        52: 11,
#     'Herbaceuous'
        71: 12,
#     'Hay/Pasture'
        81: 13,
#     'Cultivated Crops'
        82: 14,
#     'Woody Wetlands'
        90: 15,
#     'Emergent Herbaceuous Wetlands'
        95: 16
    }

def fillFlag(x):
    if x['IL_Flag'] != 1:
        return 0
    else:
        return x['IL_Flag']


def getXY(pt):
    return (pt.x, pt.y)

if __name__ == '__main__':
        
    for file in os.listdir(DATA_DIR):
            if ".gdb" in file:
                gdb_file = file
                State_name = gdb_file.split('.')[0]
                print(State_name)
                
    try:            
        parcels = gpd.read_file(DATA_DIR + gdb_file, driver='FileGDB', layer='parcels')
    except:
        parcels = gpd.read_file(DATA_DIR + gdb_file, driver='FileGDB', layer='Parcels')
            
    print('Step1: points and parcels loaded, complete')
    parcels['State_name'] = State_name
    
    parcels['Value'] = range(0,len(parcels))

    new_parcels = polygonsToRaster(parcels)
    
    print('Step2: polygon to raster, complete')
       
    NLCD  = rasterio.open(INPUT_DIR+ "NLCD_10m_NAD83.tif")
    bbox = [new_parcels.x.values.min(), new_parcels.y.values.min(), new_parcels.x.values.max(), new_parcels.y.values.max()]
    geometry = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])
    feature = [mapping(geometry)] # Required conversion
    
    id_array = new_parcels.to_array().values.astype('int_')
    
    clipped_NLCD, out_transform = mask(NLCD, feature, crop=True)
    clipped_NLCD = clipped_NLCD[0].astype('int8')
    
    clipped_NLCD = clipped_NLCD[0:id_array[0].shape[0]]
    
    del NLCD, Polygon, bbox, feature, mask, new_parcels 
    gc.collect()
    
    print('Step3: Clip NLCD, complete')
    
    idx = np.where(id_array[0].flatten() == -99999)
    
    id_array = np.delete(id_array[0].flatten(), idx)
    
    reclass_NLCD = np.delete(clipped_NLCD.flatten(), idx)
    
    del clipped_NLCD
    gc.collect()
    
    df = pd.DataFrame({'Value': id_array.flatten(), 'NLCD': reclass_NLCD.flatten()}, columns=['Value', 'NLCD'])
    
    del id_array, reclass_NLCD, idx, features, geometry
    gc.collect()
    
    print('Step4: Align ID and NLCD arrays for comparison, complete')
    
    cdf = pd.DataFrame(columns=['Value','NLCD', 'counts'])
    s = 0
    e = 10000000
    while s < len(df):
            if e > len(df):
                    e = len(df) -1
            temp = df.iloc[s:e]
            temp = temp.value_counts().rename_axis(['Value','NLCD']).reset_index(name='counts').astype('int32')
            temp['NLCD'] = temp['NLCD'].astype('int8')
            temp['Value'] = temp['Value'].astype('int32')
            temp['counts'] = temp['counts'].astype('int32')
            cdf =  pd.concat([cdf,temp])
            s += 10000000
            e += 10000000
            del temp
            gc.collect()
    del df
    gc.collect()
    
    print('Step5: Batch join arrays into dataframe, complete')
    
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
    
    df = df.iloc[:,1:].div(df.Total_Count, axis=0).drop('Total_Count', axis=1)
    
    print('Step6: Aggregate NLCD pixels per parcel ID and get total land cover counts/percentages, complete')
    
    
    counties=gpd.read_file(INPUT_DIR + 'tl_2020_us_county/tl_2020_us_county.shp')
    counties = counties.to_crs(4269)
    
    gdf = gpd.sjoin(parcels, counties, how="inner", op='within')
    gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME', 'COUNTYFP' : 'COUNTY_FIPS'})
    polygon = gdf.drop(['index_right', 'STATEFP', 'COUNTYNS', 'GEOID', 'NAMELSAD', 'LSAD', 'CLASSFP', 'MTFCC', 'CSAFP', 'CBSAFP',
                'METDIVFP', 'FUNCSTAT', 'ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON'], axis = 1)
    
    
    print('Step7: county infor written over, complete')
    
    
    #Format the indigenous lands data
    ILs = gpd.read_file(INPUT_DIR + 'Indigenous_Lands_BIA_AIAN_National_LAR.shp')
    ILs = ILs[['Name', 'geometry']]
    ILs['IL_Flag'] = 1 
    ILs = ILs.to_crs(4269)
    
    #join the parcels with indigenous lands
    polygon = gpd.sjoin(polygon, ILs, how="left", op='intersects')
    polygon['IL_Flag'] = polygon.apply(lambda x: fillFlag(x), axis = 1)
    
    print('Step8: ILs flagged, complete')
    
    
    polygon = polygon.to_crs(5070)
    Parcel_Centroid = GeoSeries(polygon['geometry']).centroid
    x,y = [list(t) for t in zip(*map(getXY, Parcel_Centroid))]
    polygon['Centroid_X'] = x
    polygon['Centroid_Y'] = y
    polygon["PARCEL_AREA"] = polygon['geometry'].area/ 10**6
    
    
    print('Step9: parcel areas calculated, complete')
    
    
    #DMP points data had duplicates that were propogating errors throughout the whole join process
    points = gpd.read_file(DATA_DIR + gdb_file, driver='FileGDB', layer='Propertypoints')
    points = points[['OWN1', 'OWN2', 
                     'MCAREOFNAM', 'MHSNUMB', 
                     'MPREDIR', 'MSTNAME', 
                     'MMODE', 'PRCLDMPID']]
    points = points.drop_duplicates(['PRCLDMPID'], keep='first')


    #there left table has to be a gdf and the right df HAS to be a simple df to result in a gdf
    state = polygon.merge(pd.DataFrame(points), on='PRCLDMPID', how= 'left')
    # state = polygon.merge(pd.DataFrame(points.drop(columns='geometry')), on='PRCLDMPID', how= 'left')
    state = pd.DataFrame(state.drop(columns='geometry'))
    
    del counties, ILs, Parcel_Centroid, gdf, parcels, points, polygon
    gc.collect()
    
    full_state = pd.merge(state, df, on='Value', how='left')
    
    col_name = {
        0:'Unclassified',
        1: 'Open Water', 
        2: 'Developed, Open Space', 
        3: 'Developed, Low Intensity',
        4: 'Developed, Medium Intensity', 
        5: 'Developed, High Intensity',
        6: 'Barren Land', 
        7: 'Deciduous Forest', 
        8: 'Evergreen Forest', 
        9: 'Mixed Forest',
        11: 'Shrub/Scrub', 
        12: 'Herbaceuous', 
        13: 'Hay/Pasture', 
        14: 'Cultivated Crops',
        15: 'Woody Wetlands', 
        16: 'Emergent Herbaceuous Wetlands',
        -99: 'Unclassified',
        19: 'Perennial Ice/Snow'
        }
    
    full_state = full_state.rename(columns=col_name)
    
    full_state.to_csv(DATA_DIR+'temp.csv')
    
    
    
    print('Preprocessing and Land Cover Analyzed')