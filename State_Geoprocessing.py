#This file spatial overlayes the DMP parcels with us census county boundaries
import os
import geopandas as gpd
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath('.'))

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    parcels = gpd.read_file(data_dir+'\\OUTPUTS\\INTERMEDIATE\\active\\'+[i for i in os.listdir(data_dir+'\\OUTPUTS\\INTERMEDIATE\\active') if i.endswith("_parcel.geojson")][0])
    state_name = parcels['State_name'].unique()[0]
    path = data_dir+'\\OUTPUTS\\INTERMEDIATE\\'+state_name
    counties=gpd.read_file(data_dir+'\\INPUTS\\tl_2020_us_county\\tl_2020_us_county.shp')
    counties = counties.to_crs(4326)
    gdf = gpd.sjoin(parcels, counties, how="inner", op='within')
    gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME', 'COUNTYFP' : 'COUNTY_FIPS'})
    polygon = gdf.drop(['index_right', 'STATEFP', 'COUNTYNS', 'GEOID', 'NAMELSAD', 'LSAD', 'CLASSFP', 'MTFCC', 'CSAFP', 'CBSAFP',
                'METDIVFP', 'FUNCSTAT', 'ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON'], axis = 1)
    #reprojecting the polygons
    polygon = polygon.to_crs(4269)
    #Format the indigenous lands data
    ILs = gpd.read_file(data_dir+'\\INPUTS\\Indigenous_Lands_BIA_AIAN_National_LAR.shp')
    ILs = ILs[['Name', 'geometry']]
    ILs['IL_Flag'] = 1 
    ILs = ILs.to_crs(4269)
    
    #join the parcels with indigenous lands
    polygon = gpd.sjoin(polygon, ILs, how="left", op='intersects')
    def fillFlag(x):
        if x['IL_Flag'] != 1:
            return 0
        else:
            return x['IL_Flag']
    polygon['IL_Flag'] = polygon.apply(lambda x: fillFlag(x), axis = 1)
    
    #calcutes the parcel centroids and the area. 
    from geopandas import GeoSeries
    def getXY(pt):
        return (pt.x, pt.y)
    points = gpd.read_file(data_dir+'\\OUTPUTS\\INTERMEDIATE\\active\\'+state_name+"_points.geojson")
    Parcel_Centroid = GeoSeries(polygon['geometry']).centroid
    x,y = [list(t) for t in zip(*map(getXY, Parcel_Centroid))]
    polygon['Centroid_X'] = x
    polygon['Centroid_Y'] = y
    polygon = polygon.to_crs(5070)
    polygon["PARCEL_AREA"] = polygon['geometry'].area/ 10**6
    #DMP points data had duplicates that were propogating errors throughout the whole join process
    points = points.drop_duplicates(['PRCLDMPID'], keep='first')
    #there left table has to be a gdf and the right df HAS to be a simple df to result in a gdf
    state = polygon.merge(pd.DataFrame(points.drop(columns='geometry')), on='PRCLDMPID', how= 'left')
    state = pd.DataFrame(state.drop(columns='geometry'))
    state.to_csv(data_dir+'\\OUTPUTS\\INTERMEDIATE\\active\\temp.csv')
