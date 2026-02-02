import os
import sys
import time
from datetime import datetime
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

pd.options.mode.chained_assignment = None

TMP_DIR = '/dev/shm'
DEBUG_DIR = TMP_DIR
KEY = 'PRCLDMPID'

# ---------- helpers ----------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_series(path, name, values):
    if values:
        pd.Series(sorted(values), name=name).to_csv(path, index=False)

def log_stage(df, label, start_ids=None):
    rows = len(df)
    if KEY in df.columns:
        ids = df[KEY].astype(str)
        uniq = ids.nunique(dropna=True)
        dup_ct = ids.duplicated(keep=False).sum()
        null_ct = df[KEY].isna().sum()
        print(f"[{now()}] [{label}] rows={rows} uniq({KEY})={uniq} dup_rows_on_{KEY}={dup_ct} null_{KEY}={null_ct}")
        dup_ids = ids[ids.duplicated(keep=False)]
        if not dup_ids.empty:
            out = os.path.join(DEBUG_DIR, f"dupes_{label}.csv")
            dup_ids.drop_duplicates().to_csv(out, index=False, header=[KEY])
            print(f"  -> wrote duplicate {KEY} list: {out} (n={dup_ids.drop_duplicates().shape[0]})")
        if start_ids is not None:
            end_ids = set(ids.dropna().astype(str))
            missing = start_ids - end_ids
            extra   = end_ids - start_ids
            print(f"  coverage_vs_start: missing={len(missing)} extra={len(extra)}")
            if missing:
                write_series(os.path.join(DEBUG_DIR, f"missing_{label}.csv"), KEY, missing)
                print(f"  -> wrote missing list: {os.path.join(DEBUG_DIR, f'missing_{label}.csv')}")
            if extra:
                write_series(os.path.join(DEBUG_DIR, f"extra_{label}.csv"), KEY, extra)
                print(f"  -> wrote extra list: {os.path.join(DEBUG_DIR, f'extra_{label}.csv')}")
    else:
        print(f"[{now()}] [{label}] rows={rows} (column {KEY} not present)")

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

# ---------- main ----------
if __name__ == '__main__':
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

    if KEY not in parcels.columns:
        raise RuntimeError(f"Expected key column '{KEY}' in parcels layer")
    start_ids = set(parcels[KEY].dropna().astype(str))
    start_rows = len(parcels)
    log_stage(parcels, "00_parcels_loaded")

    print('Step1: Points and parcels loaded')

    parcels['State_name'] = State_name
    parcels['Value'] = np.arange(len(parcels))
    new_parcels = polygonsToRaster(parcels)

    raster_path = os.path.join(TMP_DIR, f"{State_name}_Parcel_IDs.json")
    parcels[['Value', 'geometry']].to_file(raster_path, driver='GeoJSON')
    print('Step2: Polygon to raster complete')

    NLCD = rasterio.open(INPUT_DIR + "NLCD_10m_NAD83.tif")
    bbox = [new_parcels.x.values.min(), new_parcels.y.values.min(), new_parcels.x.values.max(), new_parcels.y.values.max()]
    geometry = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])
    feature = [mapping(geometry)]

    id_array = new_parcels.to_array().values.astype('int_')
    clipped_NLCD, _ = mask(NLCD, feature, crop=True)
    clipped_NLCD = clipped_NLCD[0].astype('int8')
    if clipped_NLCD.shape[0] < id_array[0].shape[0]:
        print(f"[WARN] clipped_NLCD rows < id_array rows: {clipped_NLCD.shape} vs {id_array[0].shape}")
    clipped_NLCD = clipped_NLCD[0:id_array[0].shape[0]]

    del NLCD, new_parcels
    gc.collect()
    print('Step3: Clip NLCD complete')

    idx = np.where(id_array[0].flatten() == -99999)
    id_array = np.delete(id_array[0].flatten(), idx)
    reclass_NLCD = np.delete(clipped_NLCD.flatten(), idx)

    df = pd.DataFrame({'Value': id_array.flatten(), 'NLCD': reclass_NLCD.flatten()})
    df = df[df['NLCD'] != -1]
    del id_array, reclass_NLCD, idx
    gc.collect()
    print('Step4: Align ID and NLCD complete')

    cdf = pd.DataFrame(columns=['Value', 'NLCD', 'counts'])
    s, e = 0, 10_000_000
    while s < len(df):
        if e > len(df):
            e = len(df)
        temp = df.iloc[s:e]
        temp = temp.value_counts().rename_axis(['Value', 'NLCD']).reset_index(name='counts').astype('int32')
        cdf = pd.concat([cdf, temp], ignore_index=True)
        s += 10_000_000
        e += 10_000_000
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
    print("Unique NLCD classes in state:", df.columns.tolist())
    df.columns = df.columns.to_series().map(reclass_dict)
    df['Total_Count'] = sums
    lc_cols = [c for c in df.columns if c != 'Total_Count']
    df[lc_cols] = df[lc_cols].div(df['Total_Count'], axis=0)
    df = df.drop(columns='Total_Count')

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
    gdf = gdf.set_geometry('geometry').to_crs(4269).drop(columns=['index_right'], errors='ignore')
    gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME', 'COUNTYFP': 'COUNTY_FIPS'})

    null_idx = gdf[gdf.COUNTY_NAME.isnull()].index
    for idx_i in null_idx:
        temp = gdf.loc[[idx_i]]
        temp_join = gpd.sjoin(temp, counties, how="left", predicate='intersects')
        gdf.at[idx_i, 'COUNTY_NAME'] = temp_join.NAME.values[0] if len(temp_join) else None
        gdf.at[idx_i, 'COUNTY_FIPS'] = temp_join.COUNTYFP.values[0] if len(temp_join) else None

    polygon = gdf
    log_stage(polygon, "07_after_county_sjoin", start_ids=start_ids)
    print('Step8: County info joined')

    ILs = gpd.read_file(INPUT_DIR + 'Indigenous_Lands_BIA_AIAN_National_LAR.shp')
    ILs = ILs[['Name', 'geometry']]
    ILs['IL_Flag'] = 1
    ILs = ILs.to_crs(4269)

    polygon = polygon.set_geometry('Centroid')
    polygon = gpd.sjoin(polygon, ILs, how="left", predicate='within')
    polygon['IL_Flag'] = polygon.apply(fillFlag, axis=1)
    log_stage(polygon, "08_after_IL_sjoin_pre_dedupe", start_ids=start_ids)

    before_dupe = polygon[polygon.duplicated(subset=[KEY], keep=False)][[KEY]]
    if not before_dupe.empty:
        path = os.path.join(DEBUG_DIR, "dup_rows_before_IL_drop_duplicates.csv")
        before_dupe.drop_duplicates().to_csv(path, index=False)
        print(f"  -> wrote list of duplicated {KEY} before IL drop_duplicates: {path} (n={before_dupe[KEY].nunique()})")

    polygon = polygon.drop_duplicates(subset=[KEY], keep='first')
    log_stage(polygon, "09_after_IL_drop_duplicates", start_ids=start_ids)
    print('Step9: Indigenous Lands flagged')

    # ---------- PAD overlay (patched: aggregate to 1 row/parcel) ----------
    pad = gpd.read_file(INPUT_DIR + "PAD_US_Combined.shp").to_crs(4269)
    pad = pd.concat([
        pad.loc[pad.INC_Field == 25].assign(GOV_Flag=2),  # Federal
        pad.loc[pad.INC_Field == 31].assign(GOV_Flag=3),  # State
        pad.loc[pad.INC_Field == 32].assign(GOV_Flag=4)   # Local
    ])[['GOV_Flag', "geometry"]]

    # Join: may create multiple PAD hits per parcel
    pad_join = gpd.sjoin(
        polygon.drop(columns=['index_right'], errors='ignore'),
        pad,
        how="left",
        predicate='within'
    )

    # Choose a single PAD assignment per parcel deterministically.
    # If you want FEDERAL to win over STATE over LOCAL, use categorical rank.
    # Here we prefer FEDERAL (2) > STATE (3) > LOCAL (4).
    rank_map = {2: 0, 3: 1, 4: 2, np.nan: 3}
    pad_join['_rank'] = pad_join['GOV_Flag'].map(rank_map)

    # Keep best-ranked row per parcel
    pad_best = (pad_join
                .sort_values([KEY, '_rank'])
                .drop_duplicates(subset=[KEY], keep='first'))

    # Merge the chosen GOV_Flag back (already present in pad_best)
    polygon = pad_best.drop(columns=['_rank'])

    # Apply IL precedence: keep IL_Flag==1, else use GOV_Flag (2/3/4)
    polygon['IL_Flag'] = np.where(polygon['IL_Flag'] == 1, 1, polygon['GOV_Flag'])

    log_stage(polygon, "10_after_PAD_reduce", start_ids=start_ids)
    print('Step10: PAD overlay joined')

    points = gpd.read_file(gdb_path, driver='FileGDB', layer='Propertypoints')
    points = points[['OWN1', 'OWN2', 'MCAREOFNAM', 'MHSNUMB', 'MPREDIR', 'MSTNAME', 'MMODE', KEY]]

    pre_points_dupe = points[points.duplicated(subset=[KEY], keep=False)][[KEY]]
    if not pre_points_dupe.empty:
        outp = os.path.join(DEBUG_DIR, "points_dup_parcel_ids.csv")
        pre_points_dupe.drop_duplicates().to_csv(outp, index=False)
        print(f"  -> wrote points duplicates on {KEY}: {outp} (n={pre_points_dupe[KEY].nunique()})")

    points = points.drop_duplicates([KEY], keep='first')
    log_stage(points, "11_points_after_dedupe")

    print('Step11: Ownership points loaded')

    state = polygon.merge(pd.DataFrame(points), on=KEY, how='left')
    state = pd.DataFrame(state.drop(columns='geometry', errors='ignore'))
    log_stage(state, "12_after_merge_points", start_ids=start_ids)

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
    log_stage(full_state, "13_before_write", start_ids=start_ids)

    end_rows = len(full_state)
    if end_rows != start_rows:
        end_ids = set(full_state[KEY].dropna().astype(str))
        missing = start_ids - end_ids
        extra   = end_ids - start_ids
        write_series(os.path.join(DEBUG_DIR, "final_missing_ids.csv"), KEY, missing)
        write_series(os.path.join(DEBUG_DIR, "final_extra_ids.csv"),   KEY, extra)
        raise RuntimeError(
            f"Row count mismatch: started with {start_rows}, ended with {end_rows}. "
            f"Missing={len(missing)}, Extra={len(extra)}. "
            f"See {DEBUG_DIR}/final_missing_ids.csv and final_extra_ids.csv"
        )
    else:
        print(f"[OK] Row count preserved: {end_rows} rows")

    out_path = os.path.join(TMP_DIR, "temp.csv")
    full_state.to_csv(out_path, index=False)
    print(f'✅ Preprocessing complete — Output saved to {out_path}')
