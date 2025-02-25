#!/usr/bin/env python
# coding: utf-8
import functools
import itertools

import os
import re
import sys
import time
import unicodedata
import uuid

import numpy as np
from metaphone import doublemetaphone
from tqdm.auto import tqdm
from ast import literal_eval

sys.path.insert(0, os.path.abspath('.'))

from configs import *
from utils.helpers import apply_regex, create_regex_pattern_from_dict
from utils.helpers import create_regex_pattern_from_list, parallelize_on_rows

pd.options.mode.chained_assignment = None


# Sorts the individual name combinations within the tuple and joins
# the elements back into 1 name string
def generate_normalized_name(name_tuple):
    name_arr = sorted(name_tuple)
    name_str = ''.join(name_arr)
    return name_str.upper()


def double_metaphone(value):
    return doublemetaphone(value)


def idx(x=None, y=None):
    return str(uuid.uuid4())


# @profile
def matching(inn, table_dict, unqid_dict=None, use_parallel=False):
    if use_parallel:
        meta_list, unqid, unqid_dict = inn
    else:
        meta_list = inn['Meta_names']
        unqid = inn['Unq_ID']

    # Create list of matched metaphone code
    match_list = [
        table_dict[meta_name]
        for meta_name in meta_list
        if meta_name in table_dict
    ]

    # reduce all elements matching list to a single list of place ids
    # matching all elements
    result = set(match_list[0])
    for s in match_list[1:]:
        if result.intersection(s):
            result.intersection_update(s)
    # take the reduced lists and assign each place id an unique id. note we
    # are working with place ids not the sub df's index. They don't match
    # uid = str(uuid.uuid4()) if unqid is None else unqid
    uid = str(uuid.uuid4()) if np.isnan(unqid) else unqid

    if use_parallel:
        unqid_dict.update(result, uid)
    else:
        for g in result:
            if g not in unqid_dict:
                unqid_dict[g] = uid

    return idx, uid


class Unq_ID_Obj(object):
    """Lock shared dictionary when updating (Unq_ID)"""

    def __init__(self, manager):
        self.val = manager.dict()
        self.lock = manager.Lock()

    def update(self, g_list, uid):
        # lock for update, add to dictionary if it is a new key is not yet (
        # unq_ID = None)
        with self.lock:
            for g in g_list:
                if g not in self.val:
                    self.val[g] = uid

    def value(self):
        # lock to get value
        with self.lock:
            return self.val


class Lookup_Dict_Obj(object):
    """Lock shared dictionary when updating (Lookup dictionary)"""

    def __init__(self, manager):
        self.val = manager.dict()
        self.lock = manager.Lock()

    def update(self, new_items):
        # lock for update, add to dictionary if it is a new key is not yet (
        # unq_ID = None)
        with self.lock:
            for g, person_id in new_items:
                if g in self.val:
                    hh = list(self.val[g])
                    hh.append(person_id)
                    self.val[g] = hh
                else:
                    self.val[g] = [person_id]

    def value(self):
        # lock to get value
        with self.lock:
            return self.val


# @profile
def add_combinations_to_directory(df):
    comb_tuples = df['Owners']
    meta_list = []

    for comb in comb_tuples:
        concat_name = generate_normalized_name(tuple(comb))
        metaphone_tuple = doublemetaphone(concat_name)
        meta_list.append(metaphone_tuple[0])

    return meta_list


def compute_dict(df):
    table_dict = {}
    for k_list, v in df[['Meta_names', 'place_id']].itertuples(index=False):
        for k in k_list:
            if k in table_dict:
                table_dict[k].append(v)
            else:
                table_dict[k] = [v]
    return table_dict


def vanilla_lookup(df, use_parallel=False):
    tqdm.pandas(desc='LookUp')
    df.loc[:, 'Meta_names'] = df.progress_apply(
        add_combinations_to_directory, axis=1)
    if not use_parallel:
        df_dict = compute_dict(df)
    return df if use_parallel else (df, df_dict)


def vanilla_match(df, df_dict):
    unqid_dict = {}
    tqdm.pandas(desc='Matching')
    _ = df.progress_apply(matching, axis=1, args=(df_dict, unqid_dict, False))
    return unqid_dict


def parallel_lookup(df, num_processes=10):
    df_split = np.array_split(df, num_processes)
    lookup_func = functools.partial(vanilla_lookup, use_parallel=True)

    with mp.Pool(processes=num_processes) as pool:
        ddd = pd.concat(pool.map(lookup_func, df_split))
    df_dict = compute_dict(ddd)
    return ddd, df_dict


def parallel_match(df, df_dict, num_processes=10):
    manager = mp.Manager()
    unq_id = Unq_ID_Obj(manager)

    match_func = functools.partial(matching, table_dict=df_dict,
                                   use_parallel=True)

    tuples = list(zip(df['Meta_names'].values, df['Unq_ID'].values,
                      itertools.repeat(unq_id)))

    with mp.Pool(processes=num_processes) as pool:
        _ = pool.map(match_func, tqdm(tuples, total=len(tuples)))

    return unq_id.value()


def idxx(df):
    new_df = df[1]
    new_df['new_id'] = idx()
    return new_df['new_id']


# def group_by_unq_comb_address(in_df, num_processes=100):
#     if len(in_df) <= 1:
#         return in_df
#     df = in_df[["Unq_ID", "comb_addr"]]
#     df_grouped = df.groupby(["Unq_ID", "comb_addr"])
#     # add batching mechanism here
#     with mp.Pool(num_processes) as pool:
#         desc = 'GroupBy ID Address'
#         result = pool.map(idxx, tqdm(df_grouped, desc=desc, total=len(df_grouped)))
#         # result = pool.map(idxx, tqdm(list(df_grouped), desc=desc,
#         #                              total=len(df_grouped)))
#
#     return pd.concat(result)


def group_by_unq_comb_address(in_df, num_processes=100):
    if len(in_df) <= 1:
        return in_df
    df = in_df[["Unq_ID", "comb_addr"]]
    return df.apply(lambda row: hash(tuple(row)), axis=1)


def run_one(new_df, df_dict):
    if len(new_df) > 0:
        ggg = vanilla_match(new_df, df_dict)
        new_df["Unq_ID"] = pd.Series(ggg)
    return new_df


# others
def run_other(df, df_dict, return_df=None, is_parallel=False):
    new_df = run_one(df, df_dict)
    if is_parallel:
        return_df.append(new_df)


# family
def run_family(df, df_dict, return_df=None, is_parallel=False):
    new_df = run_one(df, df_dict)
    print('\nStart Group by UnqID and Address for Family')
    new_df['Unq_ID'] = group_by_unq_comb_address(new_df,
                                                 num_processes=num_processes)
    if is_parallel:
        return_df.append(new_df)


# nans
def run_nans(df, df_dict, return_df=None, is_parallel=False):
    uuid_dict = {key: str(uuid.uuid4()) for key in df['place_id'].unique()}
    df.loc[:, 'Unq_ID'] = df.place_id.map(uuid_dict)
    if is_parallel:
        return_df.append(df)
    else:
        return df, None


# jrs
def run_jrs(df, df_dict, return_df=None, is_parallel=False):
    new_df = run_one(df, df_dict)
    print('\nStart Group by UnqID and Address for Jrs')
    new_df['Unq_ID'] = group_by_unq_comb_address(new_df)
    if is_parallel:
        return_df.append(new_df)


def run_workers(task_to_be_run, lst_df, lst_df_dict):
    # create workers

    processes = []

    # Create a manager shared list
    manager = mp.Manager()
    shared_list = manager.list()

    # spawn processes for each task
    for func, df, df_dict in zip(task_to_be_run, lst_df, lst_df_dict):
        p = mp.Process(target=func, args=(df, df_dict, shared_list, True))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    return shared_list


# def create_lookup_dict(in_df, use_parallel=False, name=''):
#     print(f'\nStart creating Lookup dictionary for {name}\n')
#     if use_parallel:
#         return parallel_lookup(in_df, num_processes=num_processes)
#     else:
#         return vanilla_lookup(in_df)


def create_lookup_dict(in_df, use_parallel=False, name=''):
    print(f'\nStart creating Lookup dictionary for {name}\n')
    len_df = len(in_df)
    if len_df == 0:
        return in_df, {}
    if use_parallel:
        if len_df < num_processes:
            num_proc = len_df
        else:
            num_proc = num_processes
        return parallel_lookup(in_df, num_processes=num_proc)
    else:
        return vanilla_lookup(in_df)


def combine_addr(MHSNUMB, MPREDIR, MSTNAME, MMODE):
    return re.sub('nan+', '', str(MHSNUMB) + ' ' + str(MPREDIR) + ' ' + str(
        MSTNAME) + ' ' + str(MMODE))


if __name__ == '__main__':

    start = time.time()

    dtype = {'Unnamed: 0': str,
             'OBJECTID_x': str,
             'PRCLDMPID': str,
             'Area': 'Float64',
             'State_name': str,
             'COUNTY_FIPS': str,
             'COUNTY_NAME': str,
             'Centroid_X': 'Float64',
             'Centroid_Y': 'Float64',
             'PARCEL_AREA': 'Float64',
             'OBJECTID_y': str,
             'OWN1': str,
             'OWN2': str,
             'MCAREOFNAM': str,
             'MHSNUMB': str,
             'MPREDIR': str,
             'MSTNAME': str,
             'MMODE': str,
             'Simple_Owners': str,
             'Owners': str,
             'place_id': 'Int64',
             'Unq_ID': str,
             'initial_class': 'Int64',
             'Own_Type': 'Float64'}

    table = pd.read_csv(DATA_DIR + input_name_matching, low_memory=False, dtype=dtype, index_col=0)

    table['Owners'] = table['Owners'].apply(literal_eval)

    table['comb_addr'] = table.apply(lambda x: combine_addr(x['MHSNUMB'], x['MPREDIR'], x['MSTNAME'], x['MMODE']),
                                     axis=1)

    # split government up further
    gov25 = table.loc[table['Own_Type'] == 25.0]
    gov31 = table.loc[table['Own_Type'] == 31.0]
    gov32 = table.loc[table['Own_Type'] == 32.0]
    family = table.loc[(table['Own_Type'] == 45.0) & (table['initial_class'] != 3)]
    jrs = table.loc[(table['Own_Type'] == 45.0) & (table['initial_class'] == 3)]
    corp = table.loc[table['Own_Type'] == 41.0]
    g42 = table.loc[table['Own_Type'] == 42.0]
    g43 = table.loc[table['Own_Type'] == 43.0]
    nans = table.loc[table['Own_Type'] == -99.0]

    gov25, __gov25_dict = create_lookup_dict(gov25, use_parallel, 'gov25')
    gov31, __gov31_dict = create_lookup_dict(gov31, use_parallel, 'gov31')
    gov32, __gov32_dict = create_lookup_dict(gov32, use_parallel, 'gov32')
    corp, __corp_dict = create_lookup_dict(corp, use_parallel, 'corp')
    g42, __g42_dict = create_lookup_dict(g42, use_parallel, 'g42')
    g43, __g43_dict = create_lookup_dict(g43, use_parallel, 'g43')  # have an address flag
    family, __family_dict = create_lookup_dict(family, use_parallel, 'Family')  # same run
    jrs, __jrs_dict = create_lookup_dict(jrs, use_parallel, 'Jrs')  # same run
    nans, __nans_dict = run_nans(nans, {})  # same run

    task_to_be_run = [run_other, run_other, run_other, run_other, run_other, run_family, run_family, run_nans, run_jrs]
    df_list = [gov25, gov31, gov32, corp, g42, g43, family, nans, jrs]
    df_dict_list = [__gov25_dict, __gov31_dict, __gov32_dict, __corp_dict, __g42_dict, __g43_dict, __family_dict,
                    __nans_dict, __jrs_dict]

    print('\nStart matching using Lookup dictionary\n')
    if use_parallel:
        final_table = pd.concat(run_workers(task_to_be_run, df_list,
                                            df_dict_list))

    else:
        for task, df, df_dict in zip(task_to_be_run, df_list, df_dict_list):
            task(df, df_dict)
        final_table = pd.concat([other, family, nans, jrs])
    final_table.sort_values(by=['PRCLDMPID'], inplace=True)

    print(f'Execution time: {time.time() - start} ')
    final_table.sort_values(by=['PRCLDMPID'], inplace=True)
    unq_count = final_table.groupby(['Unq_ID']).count().to_dict()
    final_table.loc[:, 'Total_Parcels_Owned'] = final_table.Unq_ID.map(
        unq_count['PRCLDMPID'])

    # Potentially Not delete for QC
    # if 'Meta_names' in final_table.columns:
    #     final_table.drop(columns='Meta_names', axis=1, inplace=True)

    os.remove(f'./data/new_parallel_state_temp.csv')

    # write to files
    final_table.to_csv(DATA_DIR + output_name_matching)

    print('Name Records Matched')
    print(f'time spent: {time.time() - start}')