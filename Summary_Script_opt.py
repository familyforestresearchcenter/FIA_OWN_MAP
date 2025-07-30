#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gc
import pandas as pd
from configs import *

pd.options.mode.chained_assignment = None
TMP_DIR = '/dev/shm'


def dileneate_forests(x):
    return (x['Deciduous Forest'] + x['Evergreen Forest'] +
            x['Mixed Forest'] + x['Woody Wetlands'])


def class_forest(x):
    return 1 if x['Forest_Area'] > 0.5 else 0


def forest_area(x):
    return x['Forest_Total'] * x['PARCEL_AREA'] * 247.10538146717


def reclass_own_type(x):
    if (x['Own_Type'] == 42) or (x['Own_Type'] == 43):
        return 'Other Private'
    elif x['Own_Type'] == 0:
        return 'Public'
    else:
        return str(x['Own_Type'])


if __name__ == "__main__":
    input_path = os.path.join(TMP_DIR, 'Full_Data_Table.csv')

    table = pd.read_csv(input_path, low_memory=False)

    IL = table.loc[table['IL_Flag'] == 1]
    not_IL = table.loc[table['IL_Flag'] != 1]
    IL['Own_Type'] = 44
    table = pd.concat([IL, not_IL])

    for flag_val, new_code in [(2, 25), (3, 31), (4, 32), (5, 25), (6, 31), (7, 32)]:
        flag = table.loc[table['IL_Flag'] == flag_val]
        not_flag = table.loc[table['IL_Flag'] != flag_val]
        unk = flag[flag['Own_Type'] == -99]
        known = flag[flag['Own_Type'] != -99]
        unk['Own_Type'] = new_code
        table = pd.concat([flag, not_flag, known, unk])

    gc.collect()

    table['Forest_Total'] = table.apply(dileneate_forests, axis=1)
    table['Forest_Area'] = table.apply(forest_area, axis=1)

    FApO = table.groupby('Unq_ID')['Forest_Area'].sum().to_dict()
    table['Total_Forest_Acres_Owned'] = table['Unq_ID'].map(FApO)
    table['GTR-99_Code'] = table.apply(reclass_own_type, axis=1)

    table['Total_Parcels_Owned'] = table.groupby('Unq_ID')['PRCLDMPID'].transform('count')

    table['Forest_Parcel'] = table.apply(class_forest, axis=1)
    fparc_dict = table.groupby('Unq_ID')['Forest_Parcel'].sum().to_dict()
    table['Total_Forest_Parcels_Owned'] = table['Unq_ID'].map(fparc_dict)

    state_name = table['State_name'].unique()[0]

    columns_to_drop = ['GTR-99_Code', 'State_name', 'comb_addr', 'place_id',
                       'Owners', 'Simple_Owners', 'initial_class']
    table.drop(columns=columns_to_drop, axis=1, inplace=True)

    rename_columns = {
        'PARCELAPN_x': 'PARCELAPN',
        'FIPS_x': 'FIPS',
        'Centroid_X': 'CENTROID_LONG_EPSG4269',
        'Centroid_Y': 'CENTROID_LAT_EPSG4269',
        'Own_Type': 'OWNCD',
        'Unq_ID': 'OWNERSHIP_ID',
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
        'Emergent Herbaceuous Wetlands': 'NLCD_95_PROP'
    }

    table.rename(columns=rename_columns, inplace=True)
    table['JOIN_INDEX'] = table['Value'].astype(int)

    selected_columns = ['JOIN_INDEX', 'PRCLDMPID', 'PARCEL_AREA',
                        'CENTROID_LONG_EPSG4269', 'CENTROID_LAT_EPSG4269',
                        'COUNTY_NAME', 'OWNERSHIP_ID', 'OWNCD',
                        'NLCD_11_PROP', 'NLCD_21_PROP', 'NLCD_22_PROP',
                        'NLCD_23_PROP', 'NLCD_24_PROP', 'NLCD_31_PROP',
                        'NLCD_41_PROP', 'NLCD_42_PROP', 'NLCD_43_PROP',
                        'NLCD_52_PROP', 'NLCD_71_PROP', 'NLCD_81_PROP',
                        'NLCD_82_PROP', 'NLCD_90_PROP', 'NLCD_95_PROP',
                        'Total_Parcels_Owned', 'Forest_Parcel',
                        'Total_Forest_Parcels_Owned', 'Forest_Area',
                        'Total_Forest_Acres_Owned']

    for col in selected_columns:
        if col not in table.columns:
            table[col] = 0

    table2 = table[selected_columns]

    full_output_path = os.path.join(TMP_DIR, 'Full_Data_Table.csv')
    # reduced_output_path = os.path.join(TMP_DIR, 'Reduced_Data_Table.csv')


    # todo, append owncd to the end of the unique ownership id. This will split up overlays
    # look at if unknowns share unique ids

    table.to_csv(full_output_path, index=False)
    # table2.to_csv(reduced_output_path, index=False)

    print(f'\n✅ {state_name}: Full and reduced tables written to /tmp/')
