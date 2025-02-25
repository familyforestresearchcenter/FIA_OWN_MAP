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
import xarray


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

    # os.mkdir(OUTPUT_DIR + state_name)
    try:
        os.mkdir(DATA_DIR + State_name)
    except:
        pass
    
# #     change data type for file
#     xarray.DataArray(new_parcels.to_array().values.astype('int_')[0], coords={'latitude': new_parcels['y'].values, 'longitude': new_parcels['x'].values},
#         dims=['latitude', 'longitude']).rio.to_raster(f'{DATA_DIR}{State_name}/{State_name}_Join_ID.tif', dtype=np.uint64, tiled=True, windowed=True, compress='zstd')
    
    
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


    parcels = parcels.to_crs(5070)
    Parcel_Centroid = GeoSeries(parcels['geometry']).centroid
    x,y = [list(t) for t in zip(*map(getXY, Parcel_Centroid))]
    parcels['Centroid'] = Parcel_Centroid
    parcels['Centroid_X'] = x
    parcels['Centroid_Y'] = y
    parcels["PARCEL_AREA"] = parcels['geometry'].area/ 10**6

    print("Step7: Parcel Geometric area amd centroids, complete")


    counties = counties[['COUNTYFP', 'NAME', 'geometry']]

    parcels = parcels.set_geometry('Centroid')
    parcels = parcels.to_crs(4269)

    gdf = gpd.sjoin(parcels, counties, how="left", predicate='within')

    gdf = gdf.set_geometry('geometry')
    gdf = gdf.to_crs(4269)
    gdf = gdf.drop(['index_right'], axis = 1)

    gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME', 'COUNTYFP' : 'COUNTY_FIPS'})

    for idx in gdf.loc[pd.isnull(gdf.COUNTY_NAME)].index:
        temp = gdf.loc[gdf.index == idx]
        temp_join = gpd.sjoin(temp, counties, how="left", predicate='intersects')
    # this is grabbing the first intersection
        county_name = temp_join.NAME.values[0]
        county_fip = temp_join.COUNTYFP	.values[0]
        gdf.at[idx, 'COUNTY_NAME'] = county_name
        gdf.at[idx, 'COUNTY_FIPS'] = county_fip

    polygon = gdf

# #     is within the correct choice here?
#     print('parcels: ', len(parcels))
#     gdf = gpd.sjoin(parcels, counties, how="inner", op='within')
#     print('gdf1: ', len(gdf))
#     gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME', 'COUNTYFP' : 'COUNTY_FIPS'})
#     polygon = gdf.drop(['index_right', 'STATEFP', 'COUNTYNS', 'GEOID', 'NAMELSAD', 'LSAD', 'CLASSFP', 'MTFCC', 'CSAFP', 'CBSAFP',
#                 'METDIVFP', 'FUNCSTAT', 'ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON'], axis = 1)
    
#     print('polygon1: ', len(polygon))
    
    print('Step8: County information written over, complete')
    
    
    #Format the indigenous lands data
    ILs = gpd.read_file(INPUT_DIR + 'Indigenous_Lands_BIA_AIAN_National_LAR.shp')
    ILs = ILs[['Name', 'geometry']]
    ILs['IL_Flag'] = 1 
    ILs = ILs.to_crs(4269)
    
    #join the parcels with indigenous lands

    polygon = polygon.set_geometry('Centroid')
    polygon = gpd.sjoin(polygon, ILs, how="left", predicate='within')

#     polygon = gpd.sjoin(polygon, ILs, how="left", op='intersects')
    polygon['IL_Flag'] = polygon.apply(lambda x: fillFlag(x), axis = 1)


# some centroids overlap 2 indigenous nations
    polygon = polygon.drop_duplicates(subset=['PRCLDMPID'], keep='first')


    print('Step9: ILs flagged, complete')
    

#     join the PADUS government overlay
    # pad = gpd.read_file(INPUT_DIR + "PADUS4_0_Geodatabase.gdb", driver='FileGDB', layer='PADUS4_0Combined_Proclamation_Marine_Fee_Designation_Easement')
    pad = gpd.read_file(INPUT_DIR + "PAD_US_Combined.shp")
    pad = pad.to_crs(4269)

    loc = pad.loc[pad.INC_Field == 25]
    loc['GOV_Flag'] = 2 

    stat = pad.loc[pad.INC_Field == 31]
    stat['GOV_Flag'] = 3 

    fed = pad.loc[pad.INC_Field == 32]
    fed['GOV_Flag'] = 4 

    # des = pad.loc[pad.Own_Type == 'DESG']
    # loc2 = des.loc[des.Mang_Type == 'LOC']
    # loc2['GOV_Flag'] = 5

    # stat2 = des.loc[des.Mang_Type == 'STAT']
    # stat2['GOV_Flag'] = 6

    # fed2 = des.loc[des.Mang_Type == 'FED']
    # fed2['GOV_Flag'] = 7

    pad = pd.concat([loc, stat, fed])


    # pad = pd.concat([loc, stat, fed, loc2, stat2, fed2])

    pad = pad[['GOV_Flag', "geometry"]]

    def gov_fill(x):
        if x.IL_Flag == 1:
            return x.IL_Flag
        else:
            return x.GOV_Flag

    polygon = polygon.set_geometry('Centroid')

    polygon = gpd.sjoin(polygon.drop(columns=['index_right']), pad, how="left", predicate='within')
    polygon['IL_Flag'] = polygon.apply(lambda x: gov_fill(x), axis = 1)
    print('polygon3: ', len(polygon))

    del pad, loc, stat, fed

    print('Step9: Government PAD LAyer joined, complete')

    polygon = polygon.set_geometry('geometry')


    # polygon = polygon.to_crs(5070)
    # Parcel_Centroid = GeoSeries(polygon['geometry']).centroid
    # x,y = [list(t) for t in zip(*map(getXY, Parcel_Centroid))]
    # polygon['Centroid_X'] = x
    # polygon['Centroid_Y'] = y
    # polygon["PARCEL_AREA"] = polygon['geometry'].area/ 10**6
    # print('polygon4: ', len(polygon))
    
    
    
    # print('Step10: parcel areas calculated, complete')
    
    
    #DMP points data had duplicates that were propogating errors throughout the whole join process
    points = gpd.read_file(DATA_DIR + gdb_file, driver='FileGDB', layer='Propertypoints')
    points = points[['OWN1', 'OWN2', 
                     'MCAREOFNAM', 'MHSNUMB', 
                     'MPREDIR', 'MSTNAME', 
                     'MMODE', 'PRCLDMPID']]
    points = points.drop_duplicates(['PRCLDMPID'], keep='first')
    
    
    print('polygon_pre_join: ', len(polygon))


    #there left table has to be a gdf and the right df HAS to be a simple df to result in a gdf
    state = polygon.merge(pd.DataFrame(points), on='PRCLDMPID', how= 'left')
    print('state1: ', len(state))

    # state = polygon.merge(pd.DataFrame(points.drop(columns='geometry')), on='PRCLDMPID', how= 'left')
    state = pd.DataFrame(state.drop(columns='geometry'))
    
    del counties, ILs, Parcel_Centroid, gdf, parcels, points, polygon
    gc.collect()
    
    full_state = pd.merge(state, df, on='Value', how='left')
    print('full_state1: ', len(full_state))
    

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