#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gc
import time
import pandas as pd
from configs import *

pd.options.mode.chained_assignment = None
TMP_DIR = '/dev/shm'


def now():
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def log_counts(df: pd.DataFrame, tag: str, start_ids=None, id_col: str = "PRCLDMPID", out_prefix: str = "summary"):
    rows = len(df)
    uniq = df[id_col].nunique(dropna=False)
    nulls = df[id_col].isna().sum()
    dup_rows = rows - uniq
    print(f"{now()} [{tag}] rows={rows} uniq({id_col})={uniq} dup_rows_on_{id_col}={dup_rows} null_{id_col}={nulls}")

    if start_ids is not None:
        curr_ids = set(df[id_col].astype(str))
        missing = sorted(list(start_ids - curr_ids))
        extra = sorted(list(curr_ids - start_ids))
        print(f"  coverage_vs_start: missing={len(missing)} extra={len(extra)}")
        # Write lists when present
        if missing:
            miss_path = os.path.join(TMP_DIR, f"{out_prefix}_missing_ids.csv")
            pd.Series(missing, name=id_col).to_csv(miss_path, index=False)
        if extra:
            extra_path = os.path.join(TMP_DIR, f"{out_prefix}_extra_ids.csv")
            pd.Series(extra, name=id_col).to_csv(extra_path, index=False)

    if dup_rows > 0:
        # write de-duped list of duplicates for quick inspection
        dups = (
            df[df.duplicated(id_col, keep=False)]
            .sort_values(id_col)[id_col]
            .drop_duplicates()
        )
        out_path = os.path.join(TMP_DIR, f"dupes_{tag}.csv")
        dups.to_csv(out_path, index=False)
        print(f"  -> wrote duplicate {id_col} list: {out_path} (n={len(dups)})")


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

    # Establish baseline for coverage checks
    start_ids = set(table['PRCLDMPID'].astype(str))
    log_counts(table, "00_loaded", start_ids)

    # === IL split (same behavior): set IL_Flag == 1 to 44 and recombine ===
    IL = table.loc[table['IL_Flag'] == 1]
    not_IL = table.loc[table['IL_Flag'] != 1]
    IL['Own_Type'] = 44
    table = pd.concat([IL, not_IL])
    log_counts(table, "01_after_IL_split", start_ids)

    # === PATCHED SECTION: In-place reassignment for unknowns by IL_Flag,
    #     avoiding concat that duplicated rows ===
    # Mapping flags to new Own_Type only where Own_Type == -99
    for flag_val, new_code in [(2, 25), (3, 31), (4, 32), (5, 25), (6, 31), (7, 32)]:
        mask = (table['IL_Flag'] == flag_val) & (table['Own_Type'] == -99)
        table.loc[mask, 'Own_Type'] = new_code
        log_counts(table, f"02_after_flag_{flag_val}", start_ids)

    gc.collect()
    log_counts(table, "03_after_gc", start_ids)

    # === Forest metrics ===
    table['Forest_Total'] = table.apply(dileneate_forests, axis=1)
    table['Forest_Area'] = table.apply(forest_area, axis=1)
    log_counts(table, "04_after_forest_metrics", start_ids)

    # === Group-level fields ===
    FApO = table.groupby('Unq_ID')['Forest_Area'].sum().to_dict()
    table['Total_Forest_Acres_Owned'] = table['Unq_ID'].map(FApO)
    table['GTR-99_Code'] = table.apply(reclass_own_type, axis=1)
    log_counts(table, "05_after_group_fields", start_ids)

    # === Totals by owner ===
    table['Total_Parcels_Owned'] = table.groupby('Unq_ID')['PRCLDMPID'].transform('count')
    log_counts(table, "06_after_total_parcels", start_ids)

    # === Forest parcel indicator and per-owner sum ===
    table['Forest_Parcel'] = table.apply(class_forest, axis=1)
    fparc_dict = table.groupby('Unq_ID')['Forest_Parcel'].sum().to_dict()
    table['Total_Forest_Parcels_Owned'] = table['Unq_ID'].map(fparc_dict)
    log_counts(table, "07_after_forest_parcels", start_ids)

    state_name = table['State_name'].unique()[0]

    # === Drop columns (these are non-essential for final output) ===
    columns_to_drop = ['GTR-99_Code', 'State_name', 'comb_addr', 'place_id',
                       'Owners', 'Simple_Owners', 'initial_class']
    # Guard against missing columns
    cols_existing = [c for c in columns_to_drop if c in table.columns]
    if cols_existing:
        table.drop(columns=cols_existing, axis=1, inplace=True)
    log_counts(table, "08_after_drop_cols", start_ids)

    # === Rename columns to final schema ===
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

    # JOIN_INDEX is the int version of Value
    if 'Value' in table.columns:
        table['JOIN_INDEX'] = table['Value'].astype(int)
    log_counts(table, "09_after_rename_joinindex", start_ids)

    # === Ensure final column set exists (fill absent with 0) ===
    selected_columns = [
        'JOIN_INDEX', 'PRCLDMPID', 'PARCEL_AREA',
        'CENTROID_LONG_EPSG4269', 'CENTROID_LAT_EPSG4269',
        'COUNTY_NAME', 'OWNERSHIP_ID', 'OWNCD',
        'NLCD_11_PROP', 'NLCD_21_PROP', 'NLCD_22_PROP',
        'NLCD_23_PROP', 'NLCD_24_PROP', 'NLCD_31_PROP',
        'NLCD_41_PROP', 'NLCD_42_PROP', 'NLCD_43_PROP',
        'NLCD_52_PROP', 'NLCD_71_PROP', 'NLCD_81_PROP',
        'NLCD_82_PROP', 'NLCD_90_PROP', 'NLCD_95_PROP',
        'Total_Parcels_Owned', 'Forest_Parcel',
        'Total_Forest_Parcels_Owned', 'Forest_Area',
        'Total_Forest_Acres_Owned'
    ]

    for col in selected_columns:
        if col not in table.columns:
            # Create numeric zeros for expected numeric outputs
            table[col] = 0

    # This is kept for validation; we still write the full table (as before).
    _table_for_schema_check = table[selected_columns].copy()
    log_counts(table, "10_after_selected_columns", start_ids)

    # === Write output (same as before): Full_Data_Table.csv ===
    log_counts(table, "99_before_write", start_ids)
    # Final row-count guard
    end_rows = len(table)
    start_rows = len(start_ids)
    if end_rows != start_rows:
        # Save explicit coverage diffs for quick inspection
        miss = sorted(list(start_ids - set(table['PRCLDMPID'].astype(str))))
        extra = sorted(list(set(table['PRCLDMPID'].astype(str)) - start_ids))
        pd.Series(miss, name="PRCLDMPID").to_csv(os.path.join(TMP_DIR, "summary_missing_ids.csv"), index=False)
        pd.Series(extra, name="PRCLDMPID").to_csv(os.path.join(TMP_DIR, "summary_extra_ids.csv"), index=False)
        raise RuntimeError(
            f"Row count mismatch: started with {start_rows}, ended with {end_rows}. "
            f"Missing={len(miss)}, Extra={len(extra)}. See /dev/shm/summary_missing_ids.csv and summary_extra_ids.csv"
        )
    else:
        print(f"[OK] summary row count preserved: {end_rows} rows")

    full_output_path = os.path.join(TMP_DIR, 'Full_Data_Table.csv')
    table.to_csv(full_output_path, index=False)
    print(f'\n✅ {state_name}: Full table written to /tmp/')
