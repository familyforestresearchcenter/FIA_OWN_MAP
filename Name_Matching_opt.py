import os
import time
import uuid
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import re
from configs import *
from utils.helpers import create_regex_pattern_from_dict, create_regex_pattern_from_list, apply_regex, parallelize_on_rows

TMP_DIR = '/dev/shm'
input_path = os.path.join(TMP_DIR, 'new_classified_state_temp.csv')
output_path = os.path.join(TMP_DIR, 'Full_Data_Table.csv')

pd.options.mode.chained_assignment = None

# ========= LOGGING HELPERS (behavior-neutral) =========
def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _ensure_dir(path):
    try:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass

def log_df(stage, df, start_ids=None, id_col='PRCLDMPID'):
    rows = len(df)
    uniq = df[id_col].nunique(dropna=True) if id_col in df.columns else None
    nulls = df[id_col].isna().sum() if id_col in df.columns else None
    dup_rows = (rows - uniq) if (uniq is not None) else None

    print(f"[{_ts()}] [{stage}] rows={rows}"
          + (f" uniq({id_col})={uniq}" if uniq is not None else "")
          + (f" dup_rows_on_{id_col}={dup_rows}" if dup_rows is not None else "")
          + (f" null_{id_col}={nulls}" if nulls is not None else ""))

    # Write duplicate ID list if any
    if dup_rows and dup_rows > 0 and id_col in df.columns:
        dup_ids = df[df[id_col].isin(df[id_col][df[id_col].duplicated(keep=False)])][id_col].dropna().unique()
        out = os.path.join(TMP_DIR, f"dupes_{stage}.csv")
        _ensure_dir(out)
        pd.DataFrame({id_col: dup_ids}).to_csv(out, index=False)
        print(f"  -> wrote duplicate {id_col} list: {out} (n={len(dup_ids)})")

    # Coverage vs start set (informational — does not change behavior)
    if (start_ids is not None) and (id_col in df.columns):
        cur = set(df[id_col].dropna().astype(str).values)
        missing = len(start_ids - cur)
        extra = len(cur - start_ids)
        print(f"  coverage_vs_start: missing={missing} extra={extra}")

def write_set(path, values, header='ID'):
    if values is None:
        return
    out = os.path.join(TMP_DIR, path)
    _ensure_dir(out)
    pd.DataFrame({header: sorted(values)}).to_csv(out, index=False)

# ========= ORIGINAL FUNCTIONS (unchanged) =========
def generate_normalized_name(name_tuple):
    return ''.join(sorted(name_tuple)).upper()

def double_metaphone(value):
    from metaphone import doublemetaphone
    return doublemetaphone(value)

def matching(inn, table_dict, unqid_dict=None, use_parallel=False):
    if use_parallel:
        meta_list, unqid, unqid_dict = inn
    else:
        meta_list = inn['Meta_names']
        unqid = inn['Unq_ID']

    match_list = [table_dict[m] for m in meta_list if m in table_dict]
    if not match_list:
        return str(uuid.uuid4()) if pd.isna(unqid) else unqid

    # Union all place_ids from matching Meta_names
    result = set(match_list[0])
    for s in match_list[1:]:
        result.update(s)

    # Check if any of these place_ids already have a UID
    existing_ids = {unqid_dict.get(pid) for pid in result if pid in unqid_dict}
    existing_ids.discard(None)

    # Reuse if found, otherwise generate new
    uid = sorted(existing_ids)[0] if existing_ids else (str(uuid.uuid4()) if pd.isna(unqid) else unqid)

    # Assign UID to all related place_ids (overwrite for consistency)
    for pid in result:
        unqid_dict[pid] = uid

    return uid

def add_combinations_to_directory(df):
    meta_list = []
    for comb in df['Owners']:
        concat_name = generate_normalized_name(tuple(comb))
        metaphone_tuple = double_metaphone(concat_name)
        meta_list.append(metaphone_tuple[0])
    return meta_list

def compute_dict(df):
    table_dict = {}
    for k_list, v in df[['Meta_names', 'place_id']].itertuples(index=False):
        for k in k_list:
            table_dict.setdefault(k, []).append(v)
    return table_dict

def group_by_unq_comb_address(in_df):
    if len(in_df) <= 1:
        return in_df['Unq_ID']  # return current value if only one row

    new_ids = in_df['Unq_ID'].copy()

    for uid, group in in_df.groupby("Unq_ID"):
        if group["comb_addr"].nunique() == 1:
            continue  # all addresses match, keep original UID
        else:
            # assign new UUIDs to distinct addresses within this UID
            addr_to_uuid = {addr: str(uuid.uuid4()) for addr in group["comb_addr"].unique()}
            new_ids.loc[group.index] = group["comb_addr"].map(addr_to_uuid)

    return new_ids

def combine_addr(MHSNUMB, MPREDIR, MSTNAME, MMODE):
    return re.sub('nan+', '', str(MHSNUMB) + ' ' + str(MPREDIR) + ' ' + str(MSTNAME) + ' ' + str(MMODE))

# ================== MAIN ==================
if __name__ == '__main__':
    start = time.time()

    dtype = {
        'OWN1': str, 'OWN2': str, 'MCAREOFNAM': str,
        'MHSNUMB': str, 'MPREDIR': str, 'MSTNAME': str,
        'MMODE': str, 'Simple_Owners': str, 'Owners': str,
        'place_id': 'Int64', 'Unq_ID': str, 'initial_class': 'Int64',
        'Own_Type': 'Float64', 'PRCLDMPID': str, 'State_name': str
    }

    table = pd.read_csv(input_path, dtype=dtype, low_memory=False, index_col=0)
    # Establish start-id set for coverage checks
    start_ids = set(table['PRCLDMPID'].dropna().astype(str).values)
    log_df("00_loaded", table, start_ids)

    table['Owners'] = table['Owners'].apply(literal_eval)
    table['comb_addr'] = table.apply(lambda x: combine_addr(x['MHSNUMB'], x['MPREDIR'], x['MSTNAME'], x['MMODE']), axis=1)
    log_df("01_after_parse_owners_addr", table, start_ids)

    print(table['Own_Type'].unique())

    # Split by Own_Type / class (no behavior change)
    gov25 = table.loc[table['Own_Type'] == 25.0]
    log_df("02_split_gov25", gov25, start_ids)

    gov31 = table.loc[table['Own_Type'] == 31.0]
    log_df("03_split_gov31", gov31, start_ids)

    gov32 = table.loc[table['Own_Type'] == 32.0]
    log_df("04_split_gov32", gov32, start_ids)

    family = table.loc[(table['Own_Type'] == 45.0) & (table['initial_class'] != 3)]
    log_df("05_split_family_non_jrs", family, start_ids)

    jrs = table.loc[(table['Own_Type'] == 45.0) & (table['initial_class'] == 3)]
    log_df("06_split_juniors", jrs, start_ids)

    corp = table.loc[table['Own_Type'] == 41.0]
    log_df("07_split_corp", corp, start_ids)

    g42 = table.loc[table['Own_Type'] == 42.0]
    log_df("08_split_g42", g42, start_ids)

    g43 = table.loc[table['Own_Type'] == 43.0]
    log_df("09_split_g43", g43, start_ids)

    nans = table.loc[table['Own_Type'] == -99.0]
    log_df("10_split_nans", nans, start_ids)

    # Run matching pipeline per block (unchanged behavior)
    def vanilla_lookup(df):
        df['Meta_names'] = df.apply(add_combinations_to_directory, axis=1)
        df_dict = compute_dict(df)
        return df, df_dict

    def vanilla_match(df, df_dict):
        unqid_dict = {}
        tqdm.pandas(desc='Matching')
        df['Unq_ID'] = df.progress_apply(matching, axis=1, args=(df_dict, unqid_dict, False))
        return df

    def run_block(df, label):
        if len(df) == 0:
            log_df(f"11_{label}_pre_lookup_empty", df, start_ids)
            return df
        log_df(f"11_{label}_pre_lookup", df, start_ids)
        df, df_dict = vanilla_lookup(df)
        log_df(f"12_{label}_post_lookup", df, start_ids)
        df = vanilla_match(df, df_dict)
        log_df(f"13_{label}_post_match", df, start_ids)
        return df

    def group_and_assign(df, label):
        if len(df) == 0:
            log_df(f"14_{label}_pre_group_empty", df, start_ids)
            return df
        log_df(f"14_{label}_pre_group", df, start_ids)
        df = df.reset_index(drop=True)
        df['Unq_ID'] = group_by_unq_comb_address(df)
        log_df(f"15_{label}_post_group", df, start_ids)
        return df

    gov25 = run_block(gov25, "gov25")
    gov31 = run_block(gov31, "gov31")
    gov32 = run_block(gov32, "gov32")
    corp = run_block(corp, "corp")
    g42 = run_block(g42, "g42")
    g43 = run_block(g43, "g43")
    family = run_block(family, "family")
    family = group_and_assign(family, "family")
    jrs = run_block(jrs, "jrs")
    jrs = group_and_assign(jrs, "jrs")

    # Assign Unq_ID to nans (unchanged)
    log_df("16_nans_pre_assign", nans, start_ids)
    nans['Unq_ID'] = nans['place_id'].map({pid: str(uuid.uuid4()) for pid in nans['place_id']})
    log_df("17_nans_post_assign", nans, start_ids)

    # Combine all groups
    final_table = pd.concat([gov25, gov31, gov32, corp, g42, g43, family, jrs, nans])
    log_df("18_concat_all_pre_sort", final_table, start_ids)

    # Sort and compute totals (unchanged)
    final_table.sort_values(by='PRCLDMPID', inplace=True)
    log_df("19_post_sort", final_table, start_ids)

    final_table['Total_Parcels_Owned'] = final_table.groupby('Unq_ID')['PRCLDMPID'].transform('count')
    log_df("20_post_total_parcels_owned", final_table, start_ids)

    # Final coverage check vs start_ids
    if 'PRCLDMPID' in final_table.columns:
        end_ids = set(final_table['PRCLDMPID'].dropna().astype(str).values)
        missing_ids = start_ids - end_ids
        extra_ids = end_ids - start_ids
        write_set("name_matching_missing_ids.csv", missing_ids, header='PRCLDMPID')
        write_set("name_matching_extra_ids.csv", extra_ids, header='PRCLDMPID')
        print(f"[{_ts()}] [99_before_write] rows={len(final_table)} uniq(PRCLDMPID)={final_table['PRCLDMPID'].nunique()} dup_rows_on_PRCLDMPID={len(final_table) - final_table['PRCLDMPID'].nunique()} null_PRCLDMPID={final_table['PRCLDMPID'].isna().sum()}")
        print(f"  coverage_vs_start: missing={len(missing_ids)} extra={len(extra_ids)}")
        if len(missing_ids) > 0 or len(extra_ids) > 0:
            print(f"  -> wrote name_matching_missing_ids.csv and name_matching_extra_ids.csv to {TMP_DIR}")

    # Write (unchanged)
    final_table.to_csv(output_path)
    print(f'✅ Name matching complete — saved to {output_path}')
    print(f'⏱ Execution time: {time.time() - start:.2f}s')
