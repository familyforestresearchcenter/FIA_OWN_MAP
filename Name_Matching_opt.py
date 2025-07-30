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


def generate_normalized_name(name_tuple):
    return ''.join(sorted(name_tuple)).upper()


def double_metaphone(value):
    from metaphone import doublemetaphone
    return doublemetaphone(value)


# def matching(inn, table_dict, unqid_dict=None, use_parallel=False):
#     if use_parallel:
#         meta_list, unqid, unqid_dict = inn
#     else:
#         meta_list = inn['Meta_names']
#         unqid = inn['Unq_ID']

#     match_list = [table_dict[m] for m in meta_list if m in table_dict]

#     if not match_list:
#         # uid = str(uuid.uuid4()) if pd.isna(unqid) else unqid
        
#         existing_ids = {unqid_dict.get(pid) for pid in result if pid in unqid_dict}
#         existing_ids.discard(None)

#         # Use existing ID if found, otherwise generate new one
#         if existing_ids:
#             uid = sorted(existing_ids)[0]  # pick one deterministically
#         else:
#             uid = str(uuid.uuid4()) if pd.isna(unqid) else unqid

#         return uid

#     result = set(match_list[0])
#     for s in match_list[1:]:
#         # result.intersection_update(s)
#         result.update(s)

#     uid = str(uuid.uuid4()) if pd.isna(unqid) else unqid
#     if use_parallel:
#         unqid_dict.update(result, uid)
#     else:
#         for g in result:
#             # if g not in unqid_dict:
#             #     unqid_dict[g] = uid
#             unqid_dict[g] = uid  # always override so connected records align
#     return uid

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

    # 🔥 Check if any of these place_ids already have a UID
    existing_ids = {unqid_dict.get(pid) for pid in result if pid in unqid_dict}
    existing_ids.discard(None)

    # 📌 Reuse if found, otherwise generate new
    uid = sorted(existing_ids)[0] if existing_ids else (str(uuid.uuid4()) if pd.isna(unqid) else unqid)

    # ✅ Assign UID to all related place_ids (overwrite for consistency)
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
    table['Owners'] = table['Owners'].apply(literal_eval)
    table['comb_addr'] = table.apply(lambda x: combine_addr(x['MHSNUMB'], x['MPREDIR'], x['MSTNAME'], x['MMODE']), axis=1)

    gov25 = table.loc[table['Own_Type'] == 25.0]
    gov31 = table.loc[table['Own_Type'] == 31.0]
    gov32 = table.loc[table['Own_Type'] == 32.0]
    family = table.loc[(table['Own_Type'] == 45.0) & (table['initial_class'] != 3)]
    jrs = table.loc[(table['Own_Type'] == 45.0) & (table['initial_class'] == 3)]
    corp = table.loc[table['Own_Type'] == 41.0]
    g42 = table.loc[table['Own_Type'] == 42.0]
    g43 = table.loc[table['Own_Type'] == 43.0]
    nans = table.loc[table['Own_Type'] == -99.0]

    def vanilla_lookup(df):
        df['Meta_names'] = df.apply(add_combinations_to_directory, axis=1)
        df_dict = compute_dict(df)
        return df, df_dict

    def vanilla_match(df, df_dict):
        unqid_dict = {}
        tqdm.pandas(desc='Matching')
        df['Unq_ID'] = df.progress_apply(matching, axis=1, args=(df_dict, unqid_dict, False))
        return df

    def run_block(df):
        if len(df) == 0:
            return df
        df, df_dict = vanilla_lookup(df)
        df = vanilla_match(df, df_dict)
        return df

    def group_and_assign(df):
        df = df.reset_index(drop=True)
        df['Unq_ID'] = group_by_unq_comb_address(df)
        return df



    gov25 = run_block(gov25)
    gov31 = run_block(gov31)
    gov32 = run_block(gov32)
    corp = run_block(corp)
    g42 = run_block(g42)
    g43 = run_block(g43)
    family = run_block(family)
    family = group_and_assign(family)
    jrs = run_block(jrs)
    jrs = group_and_assign(jrs)
    nans['Unq_ID'] = nans['place_id'].map({pid: str(uuid.uuid4()) for pid in nans['place_id']})

    final_table = pd.concat([gov25, gov31, gov32, corp, g42, g43, family, jrs, nans])
    final_table.sort_values(by='PRCLDMPID', inplace=True)
    final_table['Total_Parcels_Owned'] = final_table.groupby('Unq_ID')['PRCLDMPID'].transform('count')

    final_table.to_csv(output_path)
    print(f'✅ Name matching complete — saved to {output_path}')
    print(f'⏱ Execution time: {time.time() - start:.2f}s')


