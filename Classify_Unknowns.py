import os

import numpy as np
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.sparse import lil_matrix

from warnings import simplefilter

# ignore all User warnings (model compiled with older version of RandomForest)
simplefilter(action='ignore', category=UserWarning)

sys.path.insert(0, os.path.abspath('.'))

import unicodedata
import itertools

from utils.helpers import *
from configs import *
from utils.helpers import apply_regex, create_regex_pattern_from_dict
from utils.helpers import create_regex_pattern_from_list, parallelize_on_rows
import gc

pd.options.mode.chained_assignment = None


def chunks_generator(in_counts, n_keys, x_idx, y_idx, chunk_size):
    size_counts = in_counts.shape[0]
    rows = chunk_size
    cols = n_keys
    for pos in range(0, size_counts - chunk_size, chunk_size):
        curr_counts = in_counts[pos:pos + chunk_size, y_idx]
        model_sparse = lil_matrix((rows, cols))
        model_sparse[:, x_idx] = curr_counts
        yield model_sparse


def get_corp(df, wrd_list):
    wrd_regex = create_regex_pattern_from_list(wrd_list)
    selected_idx = df['Simple_Owners'].str.contains(wrd_regex)
    return df[selected_idx].copy(), df[~selected_idx]


def get_gov_row(df, wrd_list):
    wrd_regex = create_regex_pattern_from_list(wrd_list)
    selected_idx = df['OWN1'].str.contains(wrd_regex) | df[
        'OWN2'].str.contains(wrd_regex)
    return df[selected_idx].copy(), df[~selected_idx]


def preprocess_simple_owner(str1):
    str1 = str1.lower().replace(r'[^\w\s]', '')
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(x) for x in nltk.word_tokenize(str1)])


def preprocess_df_simple_owner(df):
    return df.applymap(preprocess_simple_owner)


def filEmptyStringsInOwners(x):
    if pd.isnull(x['Simple_Owners']) == True:
        if pd.isnull(x['OWN1']) == True:
            return x['Simple_Owners']
        else:
            return x['OWN1']
    else:
        return x['Simple_Owners']

def iso_biz(df):
    # create regex patterns
    wrd_srch = create_regex_pattern_from_list(keywords)
    # applying regex pattern
    wrd_df = apply_regex(df, wrd_srch)
    return (df['OWN1'].fillna('') * wrd_df['OWN1'] + ' ' + df['OWN2'].fillna('') * wrd_df['OWN2']).str.upper()


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
    init_df.loc[(nan_own1) & (~nan_own2), 'initial_class'] = -99999

    # both fields are empty
    init_df.loc[(nan_own1) & (nan_own2), 'initial_class'] = 2

    # own1 not a standard naming and OWN2 is not NaN
    init_df.loc[(non_std_naming) & (~nan_own2), 'initial_class'] = 1

    # own1 not a standard naming and OWN2 is NaN
    init_df.loc[(non_std_naming) & (nan_own2), 'initial_class'] = 0

    # if the OWN1 field is listed as an owner or as an undecided not picked
    # up by the word search
    init_df.loc[init_df['len_own1'] > 1, 'initial_class'] = 1

    init_df.loc[juniors, 'initial_class'] = 3
    init_df.loc[coorporate, 'initial_class'] = 0

    return init_df['initial_class']

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

def preprocess_all(input_data='Optimization_test_data_round_2.csv'):
    start = time.time()
    # reading and preprocessing table
    table0 = pd.read_csv(DATA_DIR + input_data,
                         index_col=0, low_memory=False)
    table0 = preprocess_table(table0)

    # Compute initial class
    table0['initial_class'] = compute_initial_class(table0[['OWN1',
                                                                'OWN2']])

    print(f'preprocessing done in : {time.time() - start}')
    return table0



if __name__ == '__main__':

    # Reading data
    # table = pd.read_csv(DATA_DIR + input_classify_unknown, index_col=0,
    #                     low_memory=False)
    
    input_classify_unknown = [i for i in os.listdir(DATA_DIR) if "temp" in i][0]

    table = preprocess_all(input_data=input_classify_unknown)

    table['Own_Type'] = None
    table['Simple_Owners'] = table.apply(lambda x: filEmptyStringsInOwners(x), axis =1)
    # null = table[table.Simple_Owners.isnull()].copy()
    null = table[table['initial_class'] == 2]
    table = table[table['initial_class'] != 2]
    null['Own_Type'] = -99

    family = table[
        (table['initial_class'] == 1) | (table['initial_class'] == 3)]
    other = table[table['initial_class'] == 0]

    # Trust
    searchfor = [' trust ', ' rev tr of ']
    trust, other = get_corp(other, searchfor)

    family_trusts, unknown_trusts = get_corp(trust, trust_kw)
    family = pd.concat([family, family_trusts])

    # ## Pass through 43 list first
    trust43, other_trust = get_corp(unknown_trusts, kw43)
    trust43['Own_Type'] = 43
    # other_trust['Own_Type'] = 41
    other = pd.concat([other, other_trust])


    # Farms
    farms, other = get_corp(other, ['farms'])
    searchfor = [' family ', ' brother ', ' son ', ' daughter ']
    family_farms, unknown_farms = get_corp(farms, searchfor)
    family = pd.concat([family, family_farms])
    other = pd.concat([other, unknown_farms])

    # Other_family_language
    
    # searchfor = [' family ', ' brother ', ' sons ', ' daughter ']
    # other_family_language, other = get_corp(other, searchfor)
    # family = pd.concat([family, other_family_language])


    #Hit Corporate Last!!
    # corporate
    # ckw = ['Rayonier', 'Weyerhaeuser', 'Plum Creek']
    corporate, other = get_corp(other, ckw)
    corporate['Own_Type'] = 41
    
    # government
    government, other = get_corp(other, government_keywords)
    unk_gov, gov = get_corp(government, government_keywords_plus)
    other = pd.concat([other, unk_gov])
    gov['Own_Type'] = 0

    # Nature conservancy
    # kw42 = ['Nature Conservancy']
    # kw42 = ['Nature Conservancy', 'Nature', 'Conservancy', 'Wild Lands', 'Wild', 'Wildlands', 'Land Trust']
    c42, other = get_corp(other, kw42)
    c42['Own_Type'] = 42

    # religious
    religious_groups, other = get_corp(other, rel_key_words)
    religious_groups['Own_Type'] = 43

    family['Own_Type'] = 45

    # Boys/Girls Scouts
    # kw43 = ['YMCA', 'YWCA', 'Boys Scouts', 'Girls Scouts']
    c43, other = get_corp(other, kw43)
    c43['Own_Type'] = 43

    # Corporate business language
    corp, other = get_corp(other, corp_keywords)
    corp['Own_Type'] = 41

    stemmer = PorterStemmer()

    # Preprocess Simple Owners
    if is_df_parallel:
        other['Simple_Owners'] = parallelize_on_rows(other[['Simple_Owners']],
                                                     preprocess_df_simple_owner)
    else:
        other['Simple_Owners'] = other.Simple_Owners.apply(
            preprocess_simple_owner)

    # Run TfidfVectorizer which is the same as running CountVectorizer followed
    # by TfidfTransformer.

    vectorizer = TfidfVectorizer()
    counts = vectorizer.fit_transform(other.Simple_Owners)

    # get feature names
    counts_key = vectorizer.get_feature_names_out()

    # loading model features
    model = load_model(DATA_DIR + 'model_dict.pkl')
    model_key = np.sort(list(model.keys()))

    # load RandomForest trained model
    classify_model = load_model(
        DATA_DIR + 'classify_unknown_ownership_model.pkl')

    # Compute intersection and indices of matching features
    keys_, x_ind, y_ind = np.intersect1d(model_key, counts_key,
                                         return_indices=True)

    # Generate prediction array and append predictions per batch
    predictions = np.array([])

    # get data batches using data generator
    #lets change chunksize to len(x_ind)
    for counts_chunk in chunks_generator(counts, model_key.shape[0], x_ind,
                                         y_ind, chunksize):
        preds = classify_model.predict(counts_chunk)
        predictions = np.append(predictions, preds, axis=0)

    # process last remaining chunk
    size_df = counts.shape[0]
    remaining_rows = size_df % chunksize
    if remaining_rows != 0:
        last_chunk_count = counts[size_df // chunksize:, y_ind]
        last_sparse = lil_matrix((remaining_rows, model_key.shape[0]))
        preds = classify_model.predict(last_sparse)
        predictions = np.append(predictions, preds, axis=0)

    # update predicted Own_Type
    other['Own_Type'] = predictions

    total = pd.concat(
        [other, trust43, family, gov, religious_groups, corporate, c42,
         c43, null, corp])

    gov = total.loc[total['Own_Type'] == 0]
    not_gov = total.loc[total['Own_Type'] != 0]


    # Add a search for the abbreviation as well
    state_name = gov['State_name'].unique()[0]
    searchfor = [' state', ' state college', ' state university', 'commonwealth', flipped_us_state[state_name], state_name]
    # print(searchfor)
    # searchfor = ['STATE ', state_name, flipped_us_state[state_name]]
    state_gov, other_gov = get_gov_row(gov, searchfor)

    
    # searchfor = ['FEDERAL ', 'BUREAU', 'UNITED STATES ']
    searchfor = federal_kw
    fed_gov, local_gov = get_gov_row(other_gov, searchfor)

    fed_gov['Own_Type'] = 25
    state_gov['Own_Type'] = 31
    local_gov['Own_Type'] = 32

    total = pd.concat([fed_gov, state_gov, local_gov, not_gov])
    
    
    # IL =  total.loc[total['IL_Flag'] == 1]
    # not_IL =  total.loc[total['IL_Flag'] != 1]
    
    # IL['Own_Type'] = 44
    # total = pd.concat([IL, not_IL])
    
    total = total.drop_duplicates(['PRCLDMPID'], keep='first')
    os.remove(f'./data/temp.csv')
    gc.collect()
    total.to_csv(DATA_DIR + output_classify_unknown)
    print('\nOwnerships Classified')
