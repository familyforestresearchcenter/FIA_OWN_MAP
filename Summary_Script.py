#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gc


sys.path.insert(0, os.path.abspath('.'))

from configs import *

pd.options.mode.chained_assignment = None


# Aggregate all relevant forest-based land cover types
def dileneate_forests(x):
    return (x['Deciduous Forest'] + x['Evergreen Forest'] + x['Mixed Forest'] +
            x['Woody Wetlands'])


def class_forest(x):
    return 1 if x['Forest_Area'] > .5 else 0


def forest_area(x):
    # return x['Forest_Total'] * x['PARCEL_AREA'] / 4047 #this equation is used to convert sq meters to acres. The Parcel Area field is in sqkm
    return x['Forest_Total'] * x['PARCEL_AREA'] * 247.10538146717 #This connverts sqkm to acres


def reclass_own_type(x):
    if (x['Own_Type'] == 42) | (x['Own_Type'] == 43):
        return 'Other Private'
    elif x['Own_Type'] == 0:
        return 'Public'
    else:
        return str(x['Own_Type'])


if __name__ == "__main__":
    # Define output directory
    # OUTPUT_DIR = DATA_DIR + 'output/'

    # Read data
    table = pd.read_csv(DATA_DIR + input_summary_script, low_memory=False)

    #this replaces the temporary variables usend in the Land_Analysis array. The variables are propoagated in the table 
    # because no data replaced them, essentially meaning they are NoData values. For now, the assumption is that 
    # these values are to be replaced with 0s. NEED CONFIRMATION

    # table = table.replace(sys.float_info.max, 0)
    
    
    IL =  table.loc[table['IL_Flag'] == 1]
    not_IL =  table.loc[table['IL_Flag'] != 1]
    
    IL['Own_Type'] = 44
    table = pd.concat([IL, not_IL])


# Check against unknown ownerships first. PAD only to fill in unknowns

    loc =  table.loc[table['IL_Flag'] == 2]
    not_loc =  table.loc[table['IL_Flag'] != 2]

    loc_unk = loc.loc[loc['Own_Type'] == -99]
    loc_k = loc.loc[loc['Own_Type'] != -99]

    loc_unk['Own_Type'] = 25
    table = pd.concat([loc, not_loc, loc_k, loc_unk])

    stat =  table.loc[table['IL_Flag'] == 3]
    not_stat =  table.loc[table['IL_Flag'] != 3]

    stat_unk = stat.loc[stat['Own_Type'] == -99]
    stat_k = stat.loc[stat['Own_Type'] != -99]
    
    stat_unk['Own_Type'] = 31
    table = pd.concat([stat, not_stat, stat_k, stat_unk])

    fed =  table.loc[table['IL_Flag'] == 4]
    not_fed =  table.loc[table['IL_Flag'] != 4]

    fed_unk = fed.loc[fed['Own_Type'] == -99]
    fed_k = fed.loc[fed['Own_Type'] != -99]
    
    fed_unk['Own_Type'] = 32
    table = pd.concat([fed, not_fed, fed_k, fed_unk])

    loc2 =  table.loc[table['IL_Flag'] == 5]
    not_loc2 =  table.loc[table['IL_Flag'] != 5]
    
    loc2_unk = loc2.loc[loc2['Own_Type'] == -99]
    loc2_k = loc2.loc[loc2['Own_Type'] != -99]

    loc2_unk['Own_Type'] = 25
    table = pd.concat([loc2, not_loc2, loc2_k, loc2_unk])

    stat2 =  table.loc[table['IL_Flag'] == 6]
    not_stat2 =  table.loc[table['IL_Flag'] != 6]

    stat2_unk = stat2.loc[stat2['Own_Type'] == -99]
    stat2_k = stat2.loc[stat2['Own_Type'] != -99]
    
    stat2_unk['Own_Type'] = 31
    table = pd.concat([stat2, not_stat2, stat2_k, stat2_unk])

    fed2 =  table.loc[table['IL_Flag'] == 7]
    not_fed2 =  table.loc[table['IL_Flag'] != 7]

    fed2_unk = fed2.loc[fed2['Own_Type'] == -99]
    fed2_k = fed2.loc[fed2['Own_Type'] != -99]
    
    fed2_unk['Own_Type'] = 32
    table = pd.concat([fed2, not_fed2, fed2_k, fed2_unk])

    del IL, loc, stat, fed, 
    not_fed, not_IL, not_loc, not_stat, 
    loc2, not_loc2, 
    stat2, not_stat2, 
    fed2, not_fed2, 
    loc2_k, loc2_unk,
    stat2_k, stat2_unk, 
    fed2_k, fed2_unk
    
    gc.collect()

    table['Forest_Total'] = table.apply(lambda x: dileneate_forests(x), axis=1)

    # Missing attribute Area , Comment next line set Forest_Area to Forest_Total
    table['Forest_Area'] = table.apply(lambda x: forest_area(x), axis=1)
    #table['Forest_Area'] = table['Forest_Total']

    # Forest Area per Owner (FApO) is the total forest area owned by all the
    # individual owner's properties Convert FApO to dictionary to map the
    # values into "Total Forested Acreas Owned"
    FApO = table.groupby(['Unq_ID']).sum()[['Forest_Area']]
    dd = FApO.to_dict()

    table['Total_Forest_Acres_Owned'] = table.Unq_ID.map(dd['Forest_Area'])
    table['GTR-99_Code'] = table.apply(lambda x: reclass_own_type(x), axis=1)

    # calculating the total forested parcels owned for each unique owner
    table['Forest_Parcel'] = table.apply(lambda x: class_forest(x), axis=1)

    fparc_dict = table.groupby(['Unq_ID']).sum()[['Forest_Parcel']].to_dict()


    #Confusing construct. May not be Useful. Consider dropping
    table['Total_Forest_Parcels_Owned'] = table.Unq_ID.map(
        fparc_dict['Forest_Parcel'])

    # table['PARCELID'] = pd.Series(range(len(table))) + ('.' + table[
    # 'FIPS_x'].astype(str)).astype(float)

    state_name = table['State_name'].unique()[0]
    #state_name = 'FAKE'

    # columns_to_drop = ['GTR-99_Code', 'OBJECTID_x',
    #                    'OBJECTID_y', 'State_name', 'comb_addr',
    #                    'place_id',
    #                    'Owners', 'Simple_Owners', 'initial_class']
    
    
    columns_to_drop = ['GTR-99_Code', 'State_name', 'comb_addr',
                       'place_id',
                       'Owners', 'Simple_Owners', 'initial_class']

    table = table.drop(columns_to_drop, axis=1)

    new_columns = {'PARCELAPN_x': 'PARCELAPN', 'FIPS_x': 'FIPS',
                   'Centroid_X': 'CENTROID_LONG_EPSG4269',
                   'Centroid_Y': 'CENTROID_LAT_EPSG4269',
                   'Own_Type': 'OWNCD', 'Unq_ID': 'OWNERSHIP_ID',
                   'Open Water': 'NLCD_11_PROP',
                   'Developed, Open Space': 'NLCD_21_PROP',
                   'Developed, Low Intensity': 'NLCD_22_PROP',
                   'Developed, Medium Intensity': 'NLCD_23_PROP',
                   'Developed, High Intensity': 'NLCD_24_PROP',
                   'Barren Land': 'NLCD_31_PROP',
                   'Deciduous Forest': 'NLCD_41_PROP',
                   'Evergreen Forest': 'NLCD_42_PROP',
                   'Mixed Forest': 'NLCD_43_PROP',
                   'Shrub/Scrub': 'NLCD_52_PROP',
                   'Herbaceuous': 'NLCD_71_PROP',
                   'Hay/Pasture': 'NLCD_81_PROP',
                   'Cultivated Crops': 'NLCD_82_PROP',
                   'Woody Wetlands': 'NLCD_90_PROP',
                   'Emergent Herbaceuous Wetlands': 'NLCD_95_PROP'}

    table = table.rename(columns=new_columns)

    # # os.mkdir(OUTPUT_DIR + state_name)
    # try:
    #     os.mkdir(DATA_DIR + state_name)
    # except:
    #     pass

    #table.to_csv(OUTPUT_DIR + state_name + '/DELAWARE_PARCELS_FULL.csv')

    os.remove(f'./data/Full_Data_Table.csv')

    table.to_csv(DATA_DIR + state_name + "//"+ state_name+'_Full_Data_Table.csv')
    
    table['JOIN_INDEX'] = table['Value'].astype(int)


    selected_columns = ['JOIN_INDEX', 'PRCLDMPID', 'PARCEL_AREA',
                        'CENTROID_LONG_EPSG4269', 'CENTROID_LAT_EPSG4269',
                        'COUNTY_NAME', 'OWNERSHIP_ID', 'OWNCD',
                        'NLCD_11_PROP', 'NLCD_21_PROP', 'NLCD_22_PROP',
                        'NLCD_23_PROP', 'NLCD_24_PROP', 'NLCD_31_PROP',
                        'NLCD_41_PROP', 'NLCD_42_PROP', 'NLCD_43_PROP',
                        'NLCD_52_PROP', 'NLCD_71_PROP',
                        'NLCD_81_PROP', 'NLCD_82_PROP', 'NLCD_90_PROP',
                        'NLCD_95_PROP', 'Total_Parcels_Owned', 'Forest_Parcel',
                        'Total_Forest_Parcels_Owned', 'Forest_Area',
                        'Total_Forest_Acres_Owned']
    
    for i in selected_columns:
        if i not in table.columns:
            table[i] = 0

    table2 = table[selected_columns]

    table2.to_csv(DATA_DIR + state_name + "//"+ state_name+'_Reduced_Data_Table.csv')

    print(f'\n{state_name}, Full and reduced tables created')
