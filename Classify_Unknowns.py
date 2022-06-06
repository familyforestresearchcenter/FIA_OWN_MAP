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

from utils.helpers import *
from configs import *

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


if __name__ == '__main__':

    # Reading data
    table = pd.read_csv(DATA_DIR + input_classify_unknown, index_col=0,
                        low_memory=False)
    table['Own_Type'] = None
    null = table[table.Simple_Owners.isnull()].copy()
    table = table[table.Simple_Owners.notnull()]
    null['Own_Type'] = 45

    family = table[
        (table['initial_class'] == 1) | (table['initial_class'] == 3)]
    other = table[table['initial_class'] == 0]

    # Trust
    trust, other = get_corp(other, trust_kw)
    searchfor = ['revocable', 'living', 'family', ' rev tr of', ' revoc trust',
                 ' revoc ']
    family_trusts, unknown_trusts = get_corp(trust, searchfor)
    family = pd.concat([family, family_trusts])
    unknown_trusts['Own_Type'] = 41

    # Farms
    farms, other = get_corp(other, ['farms'])
    searchfor = [' family ', ' brother ', ' son ', ' daughter ']
    family_farms, unknown_farms = get_corp(farms, searchfor)
    family = pd.concat([family, family_farms])
    other = pd.concat([other, unknown_farms])

    # Other_family_language
    searchfor = [' family ', ' brother ', ' sons ', ' daughter ']
    other_family_language, other = get_corp(other, searchfor)
    family = pd.concat([family, other_family_language])

    # Corporate
    corp, other = get_corp(other, corp_keywords)

    # government
    government, other = get_corp(other, government_keywords)
    unk_gov, gov = get_corp(government, government_keywords_plus)
    other = pd.concat([other, unk_gov])
    gov['Own_Type'] = 0

    # religious
    religious_groups, other = get_corp(other, rel_key_words)
    religious_groups['Own_Type'] = 43

    family['Own_Type'] = 45

    # corporate
    ckw = ['Rayonier', 'Weyerhaeuser', 'Plum Creek']
    corporate, other = get_corp(other, ckw)
    corporate['Own_Type'] = 41

    # Nature conservancy
    kw42 = ['Nature Conservancy']
    c42, other = get_corp(other, kw42)
    c42['Own_Type'] = 42

    # Boys/Girls Scouts
    kw43 = ['YMCA', 'YWCA', 'Boys Scouts', 'Girls Scouts']
    c43, other = get_corp(other, kw43)
    c43['Own_Type'] = 43

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
        [other, unknown_trusts, family, gov, religious_groups, corporate, c42,
         c43, null, corp])

    gov = total.loc[total['Own_Type'] == 0]
    not_gov = total.loc[total['Own_Type'] != 0]

    state_name = gov['State_name'].unique()[0]
    searchfor = ['STATE ', state_name, flipped_us_state[state_name]]
    state_gov, other_gov = get_gov_row(gov, searchfor)

    searchfor = ['FEDERAL ', 'BUREAU', 'UNITED STATES ']
    fed_gov, local_gov = get_gov_row(other_gov, searchfor)

    fed_gov['Own_Type'] = 25
    state_gov['Own_Type'] = 31
    local_gov['Own_Type'] = 32

    total = pd.concat([fed_gov, state_gov, local_gov, not_gov])
    total = total.drop_duplicates(['PRCLDMPID'], keep='first')
    total.to_csv(DATA_DIR + output_classify_unknown)
    print('\nOwnerships Classified')
