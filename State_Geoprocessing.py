#This file spatial overlayes the DMP parcels with us census county boundaries
import os
import geopandas as gpd
import pandas as pd

data_used_dir = r'D:\\Documents\\OwnershipMap\\New_Script\\Test_ENV\\OUTPUTS\\INTERMEDIATE\\'

parcels = gpd.read_file(data_used_dir+'active\\'+[i for i in os.listdir(data_used_dir+'active\\') if i.endswith("_parcel.geojson")][0])
#parcels = gpd.read_file(os.path.dirname(os.getcwd())+'\\OUTPUTS\\INTERMEDIATE\\active\\'+[i for i in os.listdir(os.path.dirname(os.getcwd())+'\\OUTPUTS\\INTERMEDIATE\\active') if i.endswith("_parcel.geojson")][0])
#parcels = gpd.read_file(os.path.dirname(os.getcwd())+'\\OUTPUTS\\INTERMEDIATE\\active\\'+[i for i in os.listdir(os.path.dirname(os.getcwd())+'\\OUTPUTS\\INTERMEDIATE\\active') if i.endswith("sample.geojson")][0])
state_name = parcels['State_name'].unique()[0]
path = data_used_dir+state_name
#path = os.path.dirname(os.getcwd())+'\\OUTPUTS\\INTERMEDIATE\\'+state_name
counties=gpd.read_file(r'D:\\Documents\\OwnershipMap\\New_Script\\Test_ENV\\INPUTS\\tl_2020_us_county\\tl_2020_us_county.shp')
#counties=gpd.read_file(os.path.dirname(os.getcwd())+'\\INPUTS\\tl_2020_us_county\\tl_2020_us_county.shp')
counties = counties.to_crs(4326)
gdf = gpd.sjoin(parcels, counties, how="inner", op='within')
gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME', 'COUNTYFP' : 'COUNTY_FIPS'})
polygon = gdf.drop(['index_right', 'STATEFP', 'COUNTYNS', 'GEOID', 'NAMELSAD', 'LSAD', 'CLASSFP', 'MTFCC', 'CSAFP', 'CBSAFP',
                'METDIVFP', 'FUNCSTAT', 'ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON'], axis = 1)
#gdf.to_file(path+'\\'+state_name+"_parcel_v2.geojson", driver='GeoJSON')
#calcutes the parcel centroids and the area. 
from geopandas import GeoSeries
def getXY(pt):
    return (pt.x, pt.y)
points = gpd.read_file(data_used_dir+'active\\'+state_name+"_points.geojson")
#points = gpd.read_file(os.path.dirname(os.getcwd())+'\\OUTPUTS\\INTERMEDIATE\\active\\'+state_name+"_points.geojson")
#reprojecting the polygons
polygon = polygon.to_crs(4269)
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
#state.to_csv(os.path.dirname(os.getcwd())+'\\OUTPUTS\\INTERMEDIATE\\temp.csv')
state.to_csv(r'D:\\Documents\\OwnershipMap\\New_Script\\Test_ENV\\SCRIPTS\\Vance_Working_Windows_10\\Vance_Windows\data\\temp.csv')
