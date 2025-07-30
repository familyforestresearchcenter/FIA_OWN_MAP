import os
import numpy as np
import pandas as pd
import unicodedata
import itertools
import gc
import sys
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from scipy.sparse import lil_matrix
from warnings import simplefilter
from utils.helpers import *
from configs import *
from utils.helpers import apply_regex, create_regex_pattern_from_dict, create_regex_pattern_from_list, parallelize_on_rows
import re

nltk.download('punkt')
nltk.download('punkt_tab')
pd.options.mode.chained_assignment = None
simplefilter(action='ignore', category=UserWarning)

TMP_DIR = '/dev/shm'


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


# def get_gov_row(df, wrd_list):
#     wrd_regex = create_regex_pattern_from_list(wrd_list)
#     selected_idx = df['OWN1'].str.contains(wrd_regex) | df['OWN2'].str.contains(wrd_regex)
#     return df[selected_idx].copy(), df[~selected_idx]

def get_gov_row(df, wrd_list):
    matched_rows = []
    matched_terms = []
    for kw in wrd_list:
        pattern = re.compile(kw, flags=re.IGNORECASE)
        match_idx = df['OWN1'].str.contains(pattern, na=False) | df['OWN2'].str.contains(pattern, na=False)
        if match_idx.any():
            matched_rows.append(df[match_idx])
            matched_terms.extend([kw] * match_idx.sum())
    if matched_rows:
        matched_df = pd.concat(matched_rows)
        matched_df['matched_kw'] = matched_terms
    else:
        matched_df = pd.DataFrame(columns=df.columns.tolist() + ['matched_kw'])
    unmatched_df = df.drop(index=matched_df.index, errors='ignore')
    return matched_df, unmatched_df

def preprocess_simple_owner(str1):
    str1 = str1.lower().replace(r'[^\w\s]', '')
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(x) for x in nltk.word_tokenize(str1)])


def preprocess_df_simple_owner(df):
    return df.map(preprocess_simple_owner)


def filEmptyStringsInOwners(x):
    if pd.isnull(x['Simple_Owners']):
        return x['OWN1'] if not pd.isnull(x['OWN1']) else x['Simple_Owners']
    return x['Simple_Owners']


def iso_biz(df):
    wrd_srch = create_regex_pattern_from_list(keywords)
    wrd_df = apply_regex(df, wrd_srch)
    return (df['OWN1'].fillna('') * wrd_df['OWN1'] + ' ' + df['OWN2'].fillna('') * wrd_df['OWN2']).str.upper()


def compute_initial_class(df):
    init_df = pd.DataFrame()
    wrd_srch = create_regex_pattern_from_list(keywords)
    is_junior = create_regex_pattern_from_list(junior_keywords)
    wrd_df = apply_regex(df, wrd_srch)
    juniors_df = apply_regex(df, is_junior)

    init_df['len_own1'] = df['OWN1'].fillna('').apply(lambda x: len(x.split()))
    non_std_naming = init_df['len_own1'] == 1
    nan_own2 = df['OWN2'].isnull()
    nan_own1 = df['OWN1'].isnull()

    init_df['initial_class'] = 10
    coorporate = wrd_df['OWN1'] | wrd_df['OWN2']
    juniors = juniors_df['OWN1'] | juniors_df['OWN2']

    init_df.loc[(nan_own1) & (~nan_own2), 'initial_class'] = -99999
    init_df.loc[(nan_own1) & (nan_own2), 'initial_class'] = 2
    init_df.loc[(non_std_naming) & (~nan_own2), 'initial_class'] = 1
    init_df.loc[(non_std_naming) & (nan_own2), 'initial_class'] = 0
    init_df.loc[init_df['len_own1'] > 1, 'initial_class'] = 1
    init_df.loc[juniors, 'initial_class'] = 3
    init_df.loc[coorporate, 'initial_class'] = 0

    return init_df['initial_class']


def normalize_unicode_to_ascii(data):
    val = unicodedata.normalize('NFKD', data).encode('ASCII', 'ignore').decode("utf-8")
    val = re.sub('[^A-Za-z0-9 ]+', ' ', val)
    val = re.sub(' +', ' ', val)
    return val.strip()


def generate_combinations(name_tuple):
    coms = [tuple(name_tuple)]
    if len(name_tuple) > 2:
        coms.extend(itertools.combinations(name_tuple, len(name_tuple) - 1))
    return coms


def preprocess_names(df):
    input_column = df.columns[0]
    df.replace(create_regex_pattern_from_list(NameCleaner + biz_word_drop), '', regex=True, inplace=True)
    df.replace(create_regex_pattern_from_dict(NamesExpander), inplace=True)
    df.replace(r"\b[a-zA-Z]\b", "", regex=True, inplace=True)
    df['Simple_Owners'] = df[input_column].apply(normalize_unicode_to_ascii)
    df[input_column] = df['Simple_Owners'].str.split(" ")
    return df


def preprocess_table(df0, is_other=False):
    df = pd.DataFrame()
    if is_other:
        df['Business_Name'] = iso_biz(df0).copy()
    else:
        df['Owners'] = (df0['OWN1'].fillna('') + ' ' + df0['OWN2'].fillna('')).str.upper()
    df = preprocess_names(df)

    if is_other:
        df0['Owners'] = df.Business_Name.apply(lambda x: [tuple(x)])
        return df0.reset_index(drop=True)

    df0[['Simple_Owners', 'Owners']] = df[['Simple_Owners', 'Owners']]
    df0['Owners'] = df['Owners'].apply(generate_combinations)
    df0['place_id'] = np.arange(len(df0))
    df0['Unq_ID'] = None
    return df0


def preprocess_all(input_data):
    start = time.time()
    table0 = pd.read_csv(input_data, index_col=0, low_memory=False)
    table0['OWN1'] = table0['OWN1'].replace('B L M', 'BLM')
    table0['OWN1'] = table0['OWN1'].replace('U S FOREST', 'US FOREST SERVICE')
    table0 = preprocess_table(table0)
    table0['initial_class'] = compute_initial_class(table0[['OWN1', 'OWN2']])
    print(f'Preprocessing done in: {time.time() - start:.2f}s')
    return table0


if __name__ == '__main__':
    input_path = os.path.join(TMP_DIR, "temp.csv")
    output_path = os.path.join(TMP_DIR, "new_classified_state_temp.csv")

    table = preprocess_all(input_data=input_path)
    table['Own_Type'] = None
    table['Simple_Owners'] = table.apply(filEmptyStringsInOwners, axis=1)

    # Define the keywords that indicate unavailability
    unavailable_keywords = [
        "NOT AVAILABLE FROM THE COUNTY",
        "AVAILABLE, NOT",
        "NOT AVAILABLE"
    ]

    # Identify rows with any of the unavailability keywords in OWN1 or OWN2
    not_available_county = table[
        table['OWN1'].isin(unavailable_keywords) | 
        table['OWN2'].isin(unavailable_keywords)
    ]


    # Add them to null DataFrame and set the code -99
    not_available_county['Own_Type'] = -99

    # Existing logic for initial_class == 2
    null = pd.concat([table[table['initial_class'] == 2], not_available_county])

    # Remove these records from the main table
    table = table[
        (table['initial_class'] != 2) & 
        (~table['OWN1'].isin(unavailable_keywords)) & 
        (~table['OWN2'].isin(unavailable_keywords))
    ]

    

    family = table[table['initial_class'] == 3]
    other = table[table['initial_class'].isin([0, 1])]


    trust, other = get_corp(other, [' trust ', ' rev tr of '])
    family_trusts, unknown_trusts = get_corp(trust, trust_kw)
    family = pd.concat([family, family_trusts])


    trust43, other_trust = get_corp(unknown_trusts, kw43)
    trust43['Own_Type'] = 43
    other = pd.concat([other, other_trust])


    farms, other = get_corp(other, ['farms'])
    family_farms, unknown_farms = get_corp(farms, [' family ', ' brother ', ' son ', ' daughter '])
    family = pd.concat([family, family_farms])
    other = pd.concat([other, unknown_farms])


    c42, other = get_corp(other, kw42)
    c42['Own_Type'] = 42


    religious_groups, other = get_corp(other, rel_key_words)
    religious_groups['Own_Type'] = 43


    c43, other = get_corp(other, kw43)
    
    exclusion_keywords = [r'\bGOLF\b', r'\bWORLDMARK\b']
    exclusion_pattern = '|'.join(exclusion_keywords)

    # Identify matches and split
    exclude_mask = (
        c43['OWN1'].str.contains(exclusion_pattern, case=False, na=False) |
        c43['OWN2'].str.contains(exclusion_pattern, case=False, na=False)
    )

    # Rows we keep in 43
    other = pd.concat([other, c43[exclude_mask]])
    c43 = c43[~exclude_mask]
    c43['Own_Type'] = 43


    def match_government_keywords_flexible(df, keywords_list, label=0):
        """
        Match OWN1/OWN2 using flexible token-wise matching with keyword list.
        Assigns 'Own_Type' = label where any token matches.
        """
        matched_rows = []
        for idx, row in df.iterrows():
            combined = f"{str(row['OWN1'])} {str(row['OWN2'])}".upper()
            for kw in keywords_list:
                if pd.isnull(kw):
                    continue
                if kw.strip().upper() in combined:
                    matched_rows.append(idx)
                    break
        matched_df = df.loc[matched_rows].copy()
        unmatched_df = df.drop(index=matched_rows)
        matched_df['Own_Type'] = label
        return matched_df, unmatched_df


    def create_regex_pattern_from_list2(word_list):
       return '|'.join([rf'\b{re.escape(w)}\b' for w in word_list])


    # === Step 1: Identify USA Variations FIRST ===
    usa_variations = r'\b(U(\s*\.?\s*)S(\s*\.?\s*)A(\s*\.?\s*)?)\b'

    maybe_usa = other[
        other['OWN1'].str.contains(usa_variations, flags=re.IGNORECASE, regex=True) |
        other['OWN2'].str.contains(usa_variations, flags=re.IGNORECASE, regex=True)
    ]

    # Step 1.1: Remove corporate-style names from USA matches
    def create_regex_pattern_from_literals(word_list):
        # Escapes all items, intended for plain text keywords like 'BANK'
        return '|'.join([rf'\b{re.escape(w)}\b' for w in word_list])

    def create_regex_pattern_from_raw(word_list):
        # Assumes input items are already valid regex, does not escape
        return '|'.join(word_list)

    def acronym_regex_variants(acronyms):
        # Returns both exact and flexible spacing variants for acronyms
        patterns = []
        for acr in acronyms:
            exact = rf'\b{acr}\b'
            spaced = rf'\b' + r'\s*\.?\s*'.join(list(acr)) + r'\b'
            patterns.append(exact)
            patterns.append(spaced)
        return patterns

    # Acronyms to support flexible spacing
    corp_acronyms = ['LLC', 'INC', 'CORP', 'CO', 'LTD', 'LP', 'LLP', 'PLC']

    # Compose the final pattern from both literal keywords and raw regex
    corp_filter = (
        create_regex_pattern_from_literals(
            corp_keywords + ckw + [
                'COMPANY', 'INSURANCE', 'BANK', 'MORTGAGE', 'SAVINGS',
                'FINANCIAL', 'ASSOCIATION', 'COOPERATIVE',
                'MERGENTHALER', 'HOUSING AUTHORITY', 'AFRAME PIPE'
            ]
        )
        + '|'
        + create_regex_pattern_from_raw(acronym_regex_variants(corp_acronyms))
    )

    
    corp_like_usa = (
        maybe_usa['OWN1'].str.contains(corp_filter, regex=True, case=False, na=False) |
        maybe_usa['OWN2'].str.contains(corp_filter, regex=True, case=False, na=False)
    )

    # Split true gov vs corp-like USA rows
    gov = maybe_usa[~corp_like_usa]
    usa_corp = maybe_usa[corp_like_usa]

    # Return corporate rows back to other
    other = other.drop(maybe_usa.index)
    other = pd.concat([other, usa_corp])

    # Normalize to lowercase and check substring presence
    def match_local_keywords(df, keywords):
        matched_idx = []
        for i, row in df.iterrows():
            combined = f"{row['OWN1']} {row['OWN2']}".lower()
            if any(kw.lower() in combined for kw in keywords):
                matched_idx.append(i)
        return df.loc[matched_idx], df.drop(index=matched_idx)

    # Keywords to catch local government early
    local_gov_pre_kw = [
        "city of", "town of", "village of",
        "the city of", "the town of", "city", "town", "municipal", "school district"
    ]

    likely_local_gov, other = match_local_keywords(other, local_gov_pre_kw)
    gov = pd.concat([gov, likely_local_gov], ignore_index=True)
    

    # === Step 2: Run keyword-based gov search ===
    government_keywords = list(set(government_keywords + [
        # New additions for universities and colleges
        r'\bUNIVERSITY\b',
        r'\bUNIVERSITY OF\b',
        r'\bSTATE UNIVERSITY\b',
        r'\bPUBLIC UNIVERSITY\b',
        r'\bSTATE COLLEGE\b',
        r'\bCOLLEGE OF\b',
        r'\bCOMMUNITY COLLEGE\b',
        r'\bU\.?S\.?A?\b',                             # Matches US, U.S., USA
        r'\bU\s*\.?\s*S\s*\.?\s*A?\s*\.?\b',           # Matches spaced variants: U S, U.S., U S A, etc.
        r'\bFEDERAL\b',
        r'\bCONSERVATION\b',
        r'\bGOVT\b',
        r'\bDEPARTMENT OF (AGRICULTURE|INTERIOR|DEFENSE|ENERGY|EDUCATION|TRANSPORTATION|JUSTICE|LABOR|COMMERCE)\b',
        r'\bBUREAU OF\b',
        r'\bUSDA\b',
        r'\bFOREST SERVICE\b',
        r'\bEPA\b',
        r'\bDHS\b',
        r'\bFBI\b',
        r'\bDOI\b',
        r'\bUSFS\b',
        r'\bFWS\b',
        r'\bUSFWS\b',
        r'\bDOT\b',
        r'\bUSDI\b',
        r'\bUSACE\b',
        r'\bNOAA\b',
        r'\bNPS\b',
        r'\bDOD\b',
        r'\bBLM\b',
        r'\bDOE\b',
        r'\bDOT\b',
        r'\bBIA\b',
        r'\bINTR\b',
        r'\bUSDI\b',
        r'\bB\s*L\s*M\b',
    ]))

    government, other = get_corp(other, government_keywords)

    # Step 2.1: Filter out corporate-style names again from gov matches
    # corp_like_gov = government['Simple_Owners'].str.contains(corp_filter, regex=True)
    corp_like_gov = (
        government['OWN1'].str.contains(corp_filter, regex=True, case=False, na=False) 
        #| government['OWN2'].str.contains(corp_filter, regex=True, case=False, na=False)
    )

    gov_add = government[~corp_like_gov]
    unk_gov = government[corp_like_gov]

    gov = pd.concat([gov, gov_add])

    # Put misclassified corporates back into other
    other = pd.concat([other, unk_gov])

    # Filter corp-like names mistakenly added to gov
    corp_like_gov_final = (
        gov['OWN1'].str.contains(corp_filter, regex=True, case=False, na=False)
        #| gov['OWN2'].str.contains(corp_filter, regex=True, case=False, na=False)
    )

    # Split them
    confirmed_gov = gov[~corp_like_gov_final]
    misclassified_corp = gov[corp_like_gov_final]

    # Assign gov class
    confirmed_gov['Own_Type'] = 0

    # Reassign corp-like rows back to "other"
    gov = confirmed_gov
    other = pd.concat([other, misclassified_corp], ignore_index=True)




    other_family = other[other['initial_class'] == 1]
    other = other[other['initial_class'] != 1]
    family = pd.concat([other_family, family])
    family['Own_Type'] = 45


    corp, other = get_corp(other, corp_keywords)
    corp['Own_Type'] = 41


    if is_df_parallel:
        other['Simple_Owners'] = parallelize_on_rows(other[['Simple_Owners']], preprocess_df_simple_owner)
    else:
        other['Simple_Owners'] = other['Simple_Owners'].apply(preprocess_simple_owner)

    vectorizer = TfidfVectorizer()
    counts = vectorizer.fit_transform(other.Simple_Owners)
    counts_key = vectorizer.get_feature_names_out()
    model = load_model(DATA_DIR + 'model_dict.pkl')
    model_key = np.sort(list(model.keys()))
    classify_model = load_model(DATA_DIR + 'classify_unknown_ownership_model.pkl')
    _, x_ind, y_ind = np.intersect1d(model_key, counts_key, return_indices=True)

    predictions = np.array([])
    for counts_chunk in chunks_generator(counts, model_key.shape[0], x_ind, y_ind, chunksize):
        preds = classify_model.predict(counts_chunk)
        predictions = np.append(predictions, preds)

    remaining_rows = counts.shape[0] % chunksize
    if remaining_rows != 0:
        last_sparse = lil_matrix((remaining_rows, model_key.shape[0]))
        preds = classify_model.predict(last_sparse)
        predictions = np.append(predictions, preds)

    other['Own_Type'] = predictions


    # other = other[~other.PRCLDMPID.isin(gov.PRCLDMPID)]


    total = pd.concat([
        other, trust43, family, gov, religious_groups,
        c42, c43, null, corp
    ])



    gov = total[total['Own_Type'] == 0]
    not_gov = total[total['Own_Type'] != 0]

    # Step 1: Classify Federal Government

    federal_kw = federal_kw + [ 
        r'\bU\.?S\.?A?\b',                             # Matches US, U.S., USA
        r'\bU\s*\.?\s*S\s*\.?\s*A?\s*\.?\b',           # Matches spaced variants: U S, U.S., U S A, etc.
        r'\bFEDERAL\b',
        r'\bGOVT\b',
        r'\bDEPARTMENT OF (AGRICULTURE|INTERIOR|DEFENSE|ENERGY|EDUCATION|JUSTICE|LABOR|COMMERCE)\b',
        r'\bBUREAU OF\b',
        r'\bUSDA\b',
        r'\bFOREST SERVICE\b',
        r'\bEPA\b',
        r'\bDHS\b',
        r'\bFBI\b',
        r'\bDOI\b',
        r'\bUSFS\b',
        r'\bFWS\b',
        r'\bUSFWS\b',
        r'\bUSDI\b',
        r'\bUSACE\b',
        r'\bNOAA\b',
        r'\bNPS\b',
        r'\bDOD\b',
        r'\bBLM\b',
        r'\bDOE\b',
        r'\bBIA\b',
        r'\bINTR\b',
        r'\bUSDI\b',
        r'\bB\s*L\s*M\b',
        r'\bAmerica\b'
    ]

    
    fed_gov, remaining_gov = get_gov_row(gov, federal_kw)
    # print(fed_gov[['OWN1', 'matched_kw']])
    fed_gov['Own_Type'] = 25

 # Step 2: Classify Local Government
    local_keywords = [
        r'\bCITY\b',
        r'\bTOWN\b',
        r'\bVILLAGE\b',
        r'\bCOUNTY\b',
        r'\bPARISH\b',
        r'\bBOROUGH\b',
        r'\bCOMMUNITY COLLEGE\b',
        r'\bMUNICIPAL\b',
        r'\bSCHOOL DISTRICT\b',
        r'\bFIRE DISTRICT\b',
        r'\bPOLICE DEPARTMENT\b',
        # r'\bWATER\b',
        r'\bIRRIGATION\b',
        r'\bSEWER\b',
        r'\bDRAINAGE\b',
        r'\bSANITATION\b',
        r'\Board OF (Education)\b'
    ]

    local_pattern = '|'.join(local_keywords)
    local_gov, remaining_gov = get_gov_row(remaining_gov, [local_pattern])
    local_gov['Own_Type'] = 32


    # Step 3: Classify State Government
    state_name = gov['State_name'].unique()[0].upper()
    state_keywords = [
        r'\bSTATE\b.*\b(DEPARTMENT|DEPT|UNIVERSITY|COLLEGE|OFFICE|AGENCY|AUTHORITY|SCHOOL|EDUCATION|COMMISSION)\b',
        r'\bCOMMONWEALTH\b',
        r'\bSTATE OF\b',
        r'\bSTATE OF \w+\b.*\b(DEPARTMENT|DEPT)\b',
        r'\bSTATE \w+ DEPT\b',
        r'\bDEPARTMENT\b', 
        r'\bDEPT\b',
        r'\bSTATE\b',
        r'\bDOT\b',
        r'\bDEPARTMENT OF (TRANSPORTATION)\b',
        flipped_us_state[state_name].upper(),
        state_name
    ]

    state_pattern = '|'.join(state_keywords)
    state_gov, remaining_gov = get_gov_row(remaining_gov, [state_pattern])
    state_gov['Own_Type'] = 31

    # Step 4: Any leftovers are likely local
    remaining_gov['Own_Type'] = 32

    # Step 5: Combine all
    total = pd.concat([fed_gov, local_gov, state_gov, remaining_gov, not_gov])
    total = total.drop_duplicates(['PRCLDMPID'], keep='first')


    # state_name = gov['State_name'].unique()[0]
    # state_gov, other_gov = get_gov_row(gov, [' state', ' state college', ' state university', 'commonwealth', flipped_us_state[state_name], state_name])
    # fed_gov, local_gov = get_gov_row(other_gov, federal_kw)

    # fed_gov['Own_Type'] = 25
    # state_gov['Own_Type'] = 31
    # local_gov['Own_Type'] = 32

    # total = pd.concat([fed_gov, state_gov, local_gov, not_gov])
    # total = total.drop_duplicates(['PRCLDMPID'], keep='first')

    gc.collect()
    total.to_csv(output_path)
    print(f'\n✅ Ownerships classified and saved to {output_path}')
