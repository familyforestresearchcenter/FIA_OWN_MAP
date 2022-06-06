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

sys.path.insert(0, os.path.abspath('.'))

from configs import *
from utils.helpers import apply_regex, create_regex_pattern_from_dict
from utils.helpers import create_regex_pattern_from_list, parallelize_on_rows

pd.options.mode.chained_assignment = None


def normalize_unicode_to_ascii(data):
    normal = unicodedata.normalize('NFKD', data).encode('ASCII', 'ignore')
    val = normal.decode("utf-8")
    # val = val.lower()
    # remove special characters
    val = re.sub('[^A-Za-z0-9 ]+', ' ', val)
    # remove multiple spaces
    val = re.sub(' +', ' ', val)
    return val.strip()


def generate_combinations(name_tuple):
    coms = [tuple(name_tuple)]
    i = len(name_tuple) - 1
    if i > 1:
        coms.extend(itertools.combinations(name_tuple, i))
    return coms


def preprocess_names(df):
    # Preprocess name table['Owners'] and other['Business_name']
    input_column = df.columns[0]
    df.replace(create_regex_pattern_from_list(NameCleaner + biz_word_drop), '',
               regex=True, inplace=True)
    df.replace(create_regex_pattern_from_dict(NamesExpander), inplace=True)
    # Replace single character
    df.replace(r"\b[a-zA-Z]\b", "", regex=True, inplace=True)
    df['Simple_Owners'] = df[input_column].apply(normalize_unicode_to_ascii)
    df[input_column] = df['Simple_Owners'].str.split(" ")
    return df


def preprocess_table(df0, is_other=False):
    df = pd.DataFrame()
    if is_other:
        df['Business_Name'] = iso_biz(df0).copy()
    else:
        df['Owners'] = (df0['OWN1'].fillna('') + ' ' + df0['OWN2'].fillna(
            '')).str.upper().copy()
    df = preprocess_names(df)

    # do not use generate_combinations if `other`
    if is_other:
        df0['Owners'] = df.Business_Name.apply(lambda x: [tuple(x)])
        return df0.reset_index(drop=True)

    df0[['Simple_Owners', 'Owners']] = df[['Simple_Owners', 'Owners']]
    # generate combinations
    df0['Owners'] = df['Owners'].apply(generate_combinations)

    # add place_id and Unq_ID columns
    df0['place_id'] = np.arange(len(df0))
    df0['Unq_ID'] = None
    return df0


def iso_biz(df):
    # create regex patterns
    wrd_srch = create_regex_pattern_from_list(keywords)
    # applying regex pattern
    wrd_df = apply_regex(df, wrd_srch)
    return (df['OWN1'].fillna('') * wrd_df['OWN1'] + ' ' + df['OWN2'].fillna(
        '') * wrd_df['OWN2']).str.upper()


def compute_initial_class(df):
    init_df = pd.DataFrame()

    # create regex patterns
    wrd_srch = create_regex_pattern_from_list(keywords)
    is_junior = create_regex_pattern_from_list(junior_keywords)

    # applying regex pattern
    wrd_df = apply_regex(df, wrd_srch)
    juniors_df = apply_regex(df, is_junior)

    # compute length of OWN1 
    init_df['len_own1'] = df['OWN1'].fillna('').apply(lambda x: len(x.split()))

    # not a standard naming convention length
    non_std_naming = init_df['len_own1'] == 1

    # is OWN1/OWN2 valid
    nan_own2 = df['OWN2'].isnull()
    nan_own1 = df['OWN1'].isnull()

    init_df['initial_class'] = 10

    # if ether field is labelled as another legal entity
    coorporate = wrd_df['OWN1'] | wrd_df['OWN2']

    # this will capture all juniors and run them seperately.
    juniors = juniors_df['OWN1'] | juniors_df['OWN2']

    # if field 1 is empty and field 2 is not
    init_df.loc[nan_own1 & (~nan_own2), 'initial_class'] = -99999

    # both fields are empty
    init_df.loc[nan_own1 & nan_own2, 'initial_class'] = 2

    # own1 not a standard naming and OWN2 is not NaN
    init_df.loc[non_std_naming & (~nan_own2), 'initial_class'] = 1

    # own1 not a standard naming and OWN2 is NaN
    init_df.loc[non_std_naming & nan_own2, 'initial_class'] = 0

    # if the OWN1 field is listed as an owner or as an undecided not picked
    # up by the word search
    init_df.loc[init_df['len_own1'] > 1, 'initial_class'] = 1

    init_df.loc[juniors, 'initial_class'] = 3
    init_df.loc[coorporate, 'initial_class'] = 0

    return init_df['initial_class']


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
    uid = str(uuid.uuid4()) if unqid is None else unqid

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


def group_by_unq_comb_address(in_df, num_processes=10):
    if len(in_df) <= 1:
        return in_df
    df = in_df[["Unq_ID", "comb_addr"]]
    df_grouped = df.groupby(["Unq_ID", "comb_addr"])
    with mp.Pool(num_processes) as pool:
        desc = 'GroupBy ID Address'
        result = pool.map(idxx, tqdm(list(df_grouped), desc=desc,
                                     total=len(df_grouped)))

    return pd.concat(result)


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


def create_lookup_dict(in_df, use_parallel=False, name=''):
    print(f'\nStart creating Lookup dictionary for {name}\n')
    if use_parallel:
        return parallel_lookup(in_df, num_processes=num_processes)
    else:
        return vanilla_lookup(in_df)


def preprocess_all(input_data='Optimization_test_data_round_2.csv'):
    start = time.time()
    # reading and preprocessing table
    table0 = pd.read_csv(DATA_DIR + input_data,
                         index_col=0, low_memory=False)
    table0 = preprocess_table(table0)

    # Compute initial class
    if is_df_parallel:
        table0['initial_class'] = parallelize_on_rows(table0[['OWN1', 'OWN2']],
                                                      compute_initial_class)
    else:
        table0['initial_class'] = compute_initial_class(table0[['OWN1',
                                                                'OWN2']])

    print(f'preprocessing done in : {time.time() - start}')
    return table0

def combine_addr(MHSNUMB, MPREDIR, MSTNAME, MMODE):
    return re.sub('nan+', '', str(MHSNUMB) + ' ' + str(MPREDIR) + ' ' + str(
        MSTNAME) + ' ' + str(MMODE))

if __name__ == '__main__':

    start = time.time()
    
    table = preprocess_all(input_data=input_name_matching)

    table['comb_addr'] = table.apply(lambda x: combine_addr(x['MHSNUMB'], x['MPREDIR'], x['MSTNAME'],x['MMODE']), axis=1)

    other = table.loc[table['initial_class'] == 0]
    family = table.loc[table['initial_class'] == 1]
    nans = table.loc[table['initial_class'] == 2]
    jrs = table.loc[table['initial_class'] == 3]

    other, __other_dict = create_lookup_dict(other, use_parallel, 'Other')
    family, __family_dict = create_lookup_dict(family, use_parallel, 'Family')
    jrs, __jrs_dict = create_lookup_dict(jrs, use_parallel, 'Jrs')
    nans, __nans_dict = run_nans(nans, {})

    task_to_be_run = [run_other, run_family, run_nans, run_jrs]
    df_list = [other, family, nans, jrs]

    df_dict_list = [__other_dict, __family_dict, __nans_dict, __jrs_dict]

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
        unq_count['OWN1'])

    if 'Meta_names' in final_table.columns:
        final_table.drop(columns='Meta_names', axis=1, inplace=True)

    # write to files
    final_table.to_csv(DATA_DIR + output_name_matching)

    print('Name Records Matched')
    print(f'time spent: {time.time() - start}')
