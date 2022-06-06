#!/usr/bin/env python
# coding: utf-8
import itertools
import os
import sys
import pandas as pd
import multiprocessing as mp
import time
import gc
import numpy as np
from tqdm.auto import tqdm
import rasterio

from utils.helpers import timer
from configs import DATA_DIR, idlc_batch_size, decoding_batch_size, \
    input_classified_state, input_id_dictionary, output_land_analysis, \
    input_array
from multiprocessing import Pool

pd.options.mode.chained_assignment = None

idlc = {}

sys.path.insert(0, os.path.abspath('.'))


@timer
def vectorize(func, x):
    print(f'\nExecuting {func.__name__} ...')
    vect_func = np.vectorize(func)
    return vect_func(x)


def find_ids(x):
    # return last 7 digits
    return x % int(1e7)


def find_lcs(x):
    # get two first digits
    two_digits = x // int(1e7)
    # remove trailing zero if it has one
    has_trailing_zero = two_digits % 10 == 0
    # check whether x is an array (np.ndarray) or an element of an array (
    # integer)
    if not isinstance(x, np.ndarray):
        return two_digits // 10 if has_trailing_zero else two_digits
    two_digits[has_trailing_zero] = two_digits[has_trailing_zero] // 10
    return two_digits.astype(np.uint8)


def decode_batch(x, idlc_dict):
    # find id and lc
    prclids = find_ids(x)
    lc_types = find_lcs(x)
    for prclid, lc_type in zip(prclids, lc_types):
        try:
            idlc_dict[prclid].append(lc_type)
        except KeyError:
            idlc_dict[prclid] = [lc_type]
        except Exception as e:
            raise f'Very bad, decoding error {e}' from e
    return idlc_dict


def pop_df(items):
    key, value = items
    rec = [int(key)]
    n = len(value)
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    rec.extend([value.count(num) / n for num in nums])
    return rec


def noparallel_pop_df(func, args, num_processes=8):
    result = []
    for rec in args:
        result.append(func(rec))
    # with Pool(processes=num_processes) as pool:
    #    result = list(tqdm(pool.imap(func, args), total=len(args)))
    return result


def parallel_pop_df(func, args, num_processes=8):
    with Pool(processes=num_processes) as pool:
        result = list(tqdm(pool.imap(func, args), total=len(args)))
    return result


def pop_df_1(id_key, ar_id, ar_lcs):
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    rec = [int(id_key)]
    temp_array = np.where(ar_id == id_key, 1, 0)
    test_array = ar_lcs * temp_array
    count_array = test_array[np.where(test_array != 0)].astype(str).tolist()
    rec.extend(count_array.count(num) / len(count_array) for num in nums)
    return rec


def parallel_pop_df1(func, args, ar_id, ar_lcs, num_processes=8):
    print(f'\nStart multiprocessing Pool for {func.__name__} ...')
    input_args = zip(args, itertools.repeat(ar_id, len(args)),
                     itertools.repeat(ar_lcs, len(args)))
    with Pool(processes=num_processes) as pool:
        result = pool.starmap(func, tqdm(input_args, total=len(args)),
                              chunksize=50)
    return result


def batch_process(ar_all, batch=10000):
    size_counts = len(ar_all)
    for pos in range(0, size_counts, batch):
        yield ar_all[pos:pos + batch]


def batch_dict(in_dict, batch=100000):
    size_counts = len(in_dict)
    dict0 = list(in_dict.items())
    for pos in range(0, size_counts, batch):
        yield dict0[pos:pos + batch]


import pickle


def save_dict(in_dict, filename='filename.pickle'):
    with open(filename, 'wb') as handle:
        pickle.dump(in_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_dict(filename='filename.pickle'):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


# @profile

def main(classified_state='new_classified_state_temp.csv',
         id_dictionary='ID_Dictionary.csv'):
    """
    Land Analysis Script
    input :
        - classified_state: Output of the function Classify_Unknowns_AR
        - id_dictionary   : CSV file containing JOIN_ID, PRCLDMPID pairs
    """
    # gc.set_debug(gc.DEBUG_SAVEALL)
    # Set start methods for new processes to 'spawn' which is the only
    # start method available on Windows. Unix can use 'fork' (default),
    # 'spawn' and 'forkserrver'
    # mp.set_start_method('spawn')
    table = pd.read_csv(DATA_DIR + classified_state,
                        index_col=0, low_memory=False)
    #
    table.index = table['PRCLDMPID']
    #
    
    id_df = pd.read_csv(DATA_DIR + id_dictionary)
    
    tbl_dict = dict(zip(table.PRCLDMPID, table.PRCLDMPID))
    id_dict = dict(zip(id_df.Value, id_df.PRCLDMPID))
    id_dict = {key:val for key, val in id_dict.items() if val in tbl_dict}
    #id_dict = dict(zip(id_df.Value, id_df.PRCLDMPID))

    #ar_df = pd.read_csv(DATA_DIR + input_array, names=['ar'],
                        #dtype=np.uint32, low_memory=False)



    #ar = ar_df['ar'].values

                        
    ds = rasterio.open(DATA_DIR + input_array)
    ar =ds.read(1)
    ar = ar.flatten()
    ar =ar.astype('int32')

    #ar = ar[:100000]
               

    general_values = [100000000, 110000000, 120000000, 130000000, 140000000,
                      150000000, 160000000, 200000000, 300000000, 400000000,
                      500000000, 600000000, 700000000, 800000000, 900000000]

    mask_general_values = np.in1d(ar, general_values, invert=True)
    ar_general_public = ar[~mask_general_values].copy()

    new_nums = list(np.unique(ar_general_public))[1:]
    gp = ['General Public']
    gp.extend(
        np.count_nonzero(ar_general_public == n) / len(ar_general_public)
        for n in new_nums
    )

    ar = ar[mask_general_values]
    ar = ar[np.where(ar != -2147483647)]

    idlc = {}
    start = time.time()
    print('\nDecoding arrays and creating idlc dictionary ...')
    n_batch = (len(ar) // decoding_batch_size) + 1
    for i, ar_batch in enumerate(batch_process(ar, decoding_batch_size)):
        print('', end=f'\rProcessing {i + 1:2d} of {n_batch} batches')
        idlc = decode_batch(ar_batch, idlc)
    print(f'\nDecoding done in {time.time() - start} s')

    df_columns = ['JOIN_INDEX', 'Open Water', 'Developed, Open Space',
                  'Developed, Low Intensity', 'Developed, Medium Intensity',
                  'Developed, High Intensity', 'Barren Land',
                  'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest',
                  'Shrub/Scrub', 'Herbaceuous', 'Hay/Pasture',
                  'Cultivated Crops', 'Woody Wetlands',
                  'Emergent Herbaceuous Wetlands']
    
    #removes ids that are not shared between dictionary and table
    idlc = {key:val for key, val in idlc.items() if key in id_dict}
    #del idlc['000000']
    
    table[df_columns] = sys.float_info.max
    table['JOIN_INDEX'] = table['JOIN_INDEX'].astype(int)
    del ar, ds, ar_batch, id_df, ar_general_public
    gc.collect()
    n_batch = (len(idlc.keys()) // idlc_batch_size) + 1

    for i, idlc_batch in enumerate(batch_dict(idlc, batch=idlc_batch_size)):
        print(f'\nStart Multiprocessing Pool: {i + 1} of {n_batch} batches')
        start = time.time()
        # res_pop_df = create_fake_res_pop(idlc_batch)
        res_pop_df = parallel_pop_df(pop_df, idlc_batch)
        # res_pop_df = noparallel_pop_df(pop_df, idlc_batch)
        prcldmpid = list(map(lambda x: id_dict[x[0]], res_pop_df))
        print(prcldmpid, res_pop_df)
        table.loc[prcldmpid, df_columns] = res_pop_df
        del idlc_batch, res_pop_df, prcldmpid
        gc.collect()
        print(f'Done in {time.time() - start} s')

    table.to_csv(DATA_DIR + output_land_analysis)

    print('\nLand Cover Analyzed')


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    main(classified_state=input_classified_state,
         id_dictionary=input_id_dictionary)
