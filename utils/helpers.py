import re
import pandas as pd
import numpy as np
import multiprocessing as mp
import pickle
import time
from tqdm.auto import tqdm


nums = [str(i) for i in range(1, 17)]
nums.remove('10')

def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Finished in {(t2-t1):.4f}s')
        return result
    return wrap_func


def run_parallel_for_loop(func, args, num_processes=8):
    print(f'\nStart multiprocessing Pool for {func.__name__} ...')
    with mp.Pool(processes=num_processes) as pool:
        result = list(tqdm(pool.imap(func, args, total=len(args))))
    return result


def parallelize_on_rows(data, func, num_of_processes=8):
    df_split = np.array_split(data, num_of_processes)
    with mp.Pool(num_of_processes) as p:
        df = pd.concat(p.map(func, df_split))
    return df


def create_fakedata(n_rows):
    columns = ['PRCLDMPID', 'OWN1', 'OWN2', 'comb_addr', 'State_name',
               'Simple_Owners', 'Owners', 'place_id', 'Unq_ID',
               'initial_class', 'Total_Parcels_Owned', 'Own_Type']
    df = pd.DataFrame(columns=columns)
    data_prototype = [1063, 'Chung Group LLC', 'Jennifer Waters',
                      '18145 Sean Road Apt. 338', 'FAKE',
                      'chung group jennif water', "[('CHUNG', 'GROUP', "
                                                  "'JENNIFER', 'WATERS'), "
                                                  "('CHUNG', 'GROUP', 'JENNIFER'), ('CHUNG', 'GROUP', 'WATERS'), ('CHUNG', 'JENNIFER', 'WATERS'), ('GROUP', 'JENNIFER', 'WATERS')]",
                      1063, '4b84ef19-2d45-4356-b0a5-22b363019f13', 0, 1, 32.0]
    prcldmpid = np.arange(n_rows)
    np.random.shuffle(prcldmpid)
    df['PRCLDMPID'] = prcldmpid
    for k, v in zip(columns[1:], data_prototype[1:]):
        df[k] = v

    id_df = dict(zip(prcldmpid, np.arange(n_rows)))
    return df, id_df


def create_fake_res_pop(idlc_batch):
    n_rows = len(idlc_batch)
    ff = np.random.dirichlet(np.ones(15), size=n_rows)
    join_index = np.array(list(dict(idlc_batch).keys()))
    return np.hstack([join_index.reshape(n_rows, 1), ff]).tolist()


def create_idlc(k):
    dd = {}
    for i in k:
        dd[i] = np.random.randint(0, 15, size=100).tolist()
    return dd


def apply_regex(df, regex_pattern, columns=None):
    if columns is None:
        columns = ['OWN1', 'OWN2']
    # Applying regex pattern
    out_df = pd.DataFrame()
    for col in columns:
        out_df[col] = df[col].fillna('').str.contains(
            regex_pattern, regex=True)
    return out_df


def create_regex_pattern_from_list(wrd_list):
    # Given a list of words, This function create a regex pattern
    joined_keys = '|'.join([v.strip() for v in wrd_list])
    return re.compile(r'\b(?:' + joined_keys + r')\b', flags=re.I)


def create_regex_pattern_from_dict(wrd_dict):
    # Given a list of words, This function create a regex pattern
    dict_item = wrd_dict.items()
    return {r'(\b){}(\b)'.format(k): r'\1{}\2'.format(v) for k, v in dict_item}


def load_model(model_pickle):
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)
    return model

