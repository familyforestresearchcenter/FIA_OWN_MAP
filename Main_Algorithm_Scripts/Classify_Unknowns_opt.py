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
import time  # added (used in preprocess_all)

# -------------------- LOGGING HELPERS (added; no behavior change) --------------------
from datetime import datetime

DEBUG_DIR = '/dev/shm'
KEY = 'PRCLDMPID'  # primary key to track coverage

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _write_series(path, name, values):
    if values:
        pd.Series(sorted(values), name=name).to_csv(path, index=False)

def log_totals(df, label, start_ids=None, write_dupes=True):
    """
    Prints rows, unique PRCLDMPID, duplicate rows on PRCLDMPID, and nulls.
    If start_ids provided, also prints coverage vs baseline and writes CSVs for missing/extra IDs.
    """
    rows = len(df)
    if KEY in df.columns:
        ids = df[KEY].astype(str)
        uniq = ids.nunique(dropna=True)
        dup_ct = ids.duplicated(keep=False).sum()
        null_ct = df[KEY].isna().sum()
        print(f"[{_now()}] [{label}] rows={rows} uniq({KEY})={uniq} dup_rows_on_{KEY}={dup_ct} null_{KEY}={null_ct}")
        if write_dupes and dup_ct:
            dup_ids = ids[ids.duplicated(keep=False)].drop_duplicates()
            out = os.path.join(DEBUG_DIR, f"dupes_{label}.csv")
            dup_ids.to_csv(out, index=False, header=[KEY])
            print(f"  -> wrote duplicate {KEY} list: {out} (n={dup_ids.shape[0]})")
        if start_ids is not None:
            end_ids = set(ids.dropna().astype(str))
            missing = start_ids - end_ids
            extra   = end_ids - start_ids
            print(f"  coverage_vs_start: missing={len(missing)} extra={len(extra)}")
            if missing:
                _write_series(os.path.join(DEBUG_DIR, f"missing_{label}.csv"), KEY, missing)
            if extra:
                _write_series(os.path.join(DEBUG_DIR, f"extra_{label}.csv"),   KEY, extra)
    else:
        print(f"[{_now()}] [{label}] rows={rows} (no '{KEY}' column)")
# -------------------- END LOGGING HELPERS --------------------

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
    unmatched_df = df[~df[KEY].isin(matched_df[KEY])]
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
    table0.loc[table0['initial_class'] == -99999, 'initial_class'] = 1
    print(f'Preprocessing done in: {time.time() - start:.2f}s')
    return table0


MISSING_ID = "100660192_89848561"

def dbg(df, label):
    """Small debug block to track the missing record."""
    if KEY not in df.columns:
        print(f"[DEBUG:{label}] No PRCLDMPID column")
        return
    present = MISSING_ID in set(df[KEY].astype(str))
    print(f"[DEBUG:{label}] rows={len(df):,}  present={present}")
    if not present:
        print(f"⚠️  Missing after step: {label}")


if __name__ == '__main__':
    input_path = os.path.join(TMP_DIR, "temp.csv")
    output_path = os.path.join(TMP_DIR, "new_classified_state_temp.csv")

    table = preprocess_all(input_data=input_path)

    # --- BASELINE LOG ---
    if KEY not in table.columns:
        raise RuntimeError(f"Expected key column '{KEY}' in input")
    _start_ids = set(table[KEY].dropna().astype(str))
    _start_rows = len(table)

    log_totals(table, "00_loaded")
    dbg(table, "00_loaded")

    table['Own_Type'] = None
    table['Simple_Owners'] = table.apply(filEmptyStringsInOwners, axis=1)
    log_totals(table, "01_after_simple_owners_fill")
    dbg(table, "01_after_simple_owners_fill")

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
    log_totals(not_available_county, "02_not_available_bucket")
    dbg(not_available_county, "02_not_available_bucket")

    # Add them to null DataFrame and set the code -99
    not_available_county['Own_Type'] = -99

    # Existing logic for initial_class == 2
    null = pd.concat([table[table['initial_class'] == 2], not_available_county])
    null['Own_Type'] = -99
    log_totals(null, "03_null_bucket")
    dbg(null, "03_null_bucket")

    # Remove these records from the main table
    table = table[
        (table['initial_class'] != 2) & 
        (~table['OWN1'].isin(unavailable_keywords)) & 
        (~table['OWN2'].isin(unavailable_keywords))
    ]
    log_totals(table, "04_after_remove_null_unavailable")
    dbg(table, "04_after_remove_null_unavailable")
    dbg(table, "04_after_remove_null_unavailable")

    # Split to family/other
    family = table[table['initial_class'] == 3]
    other = table[table['initial_class'].isin([0, 1])]
    log_totals(family, "05_family_initial")
    dbg(family, "05_family_initial")

    log_totals(other,  "06_other_initial")
    dbg(other, "06_other_initial")

    # Trusts
    trust, other = get_corp(other, [' trust ', ' rev tr of '])
    family_trusts, unknown_trusts = get_corp(trust, trust_kw)
    family = pd.concat([family, family_trusts])
    log_totals(family,         "07_family_after_family_trusts")
    dbg(family, "07_family_after_family_trusts")
    log_totals(unknown_trusts, "08_unknown_trusts")
    dbg(unknown_trusts, "08_unknown_trusts")
    log_totals(other,          "09_other_after_trust_split")
    dbg(other, "09_other_after_trust_split")

    trust43, other_trust = get_corp(unknown_trusts, kw43)
    trust43['Own_Type'] = 43
    other = pd.concat([other, other_trust])
    log_totals(trust43, "10_trust43_bucket")
    log_totals(other,   "11_other_after_unknown_trusts")

    # Farms
    farms, other = get_corp(other, ['farms'])
    family_farms, unknown_farms = get_corp(farms, [' family ', ' brother ', ' son ', ' daughter '])
    family = pd.concat([family, family_farms])
    other = pd.concat([other, unknown_farms])
    log_totals(family, "12_family_after_farms")
    log_totals(other,  "13_other_after_farms")

    # 42 / religious / 43 corp
    c42, other = get_corp(other, kw42)
    c42['Own_Type'] = 42
    log_totals(c42,   "14_c42_bucket")
    dbg(c42, "14_c42_bucket")
    log_totals(other, "15_other_after_c42")
    dbg(other, "15_other_after_c42")

    religious_groups, other = get_corp(other, rel_key_words)
    religious_groups['Own_Type'] = 43
    log_totals(religious_groups, "16_religious_groups_bucket")
    dbg(religious_groups, "16_religious_groups_bucket")
    log_totals(other,            "17_other_after_religious")
    dbg(other, "17_other_after_religious")

    c43, other = get_corp(other, kw43)
    exclusion_keywords = [r'\bGOLF\b', r'\bWORLDMARK\b']
    exclusion_pattern = '|'.join(exclusion_keywords)
    exclude_mask = (
        c43['OWN1'].str.contains(exclusion_pattern, case=False, na=False) |
        c43['OWN2'].str.contains(exclusion_pattern, case=False, na=False)
    )
    other = pd.concat([other, c43[exclude_mask]])
    c43 = c43[~exclude_mask]
    c43['Own_Type'] = 43
    log_totals(c43,   "18_c43_bucket")
    dbg(c43, "18_c43_bucket")
    log_totals(other, "19_other_after_c43_exclusions")
    dbg(other, "19_other_after_c43_exclusions")


    # === Step 1: Identify USA Variations FIRST ===
    usa_variations = r'\b(U(\s*\.?\s*)S(\s*\.?\s*)A(\s*\.?\s*)?)\b'
    maybe_usa = other[
        other['OWN1'].str.contains(usa_variations, flags=re.IGNORECASE, regex=True) |
        other['OWN2'].str.contains(usa_variations, flags=re.IGNORECASE, regex=True)
    ]
    log_totals(maybe_usa, "20_maybe_usa")
    dbg(maybe_usa, "20_maybe_usa")

    # Step 1.1: Remove corporate-style names from USA matches
    def create_regex_pattern_from_literals(word_list):
        return '|'.join([rf'\b{re.escape(w)}\b' for w in word_list])

    def create_regex_pattern_from_raw(word_list):
        return '|'.join(word_list)

    def acronym_regex_variants(acronyms):
        patterns = []
        for acr in acronyms:
            exact = rf'\b{acr}\b'
            spaced = rf'\b' + r'\s*\.?\s*'.join(list(acr)) + r'\b'
            patterns.append(exact)
            patterns.append(spaced)
        return patterns

    corp_acronyms = ['LLC', 'INC', 'CORP', 'CO', 'LTD', 'LP', 'LLP', 'PLC']
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
    gov = maybe_usa[~corp_like_usa]
    usa_corp = maybe_usa[corp_like_usa]

    other = other[~other[KEY].isin(maybe_usa[KEY])]
    other = pd.concat([other, usa_corp])
    log_totals(gov,   "21_gov_from_maybe_usa")
    dbg(gov, "21_gov_from_maybe_usa")
    log_totals(other, "22_other_after_maybe_usa_split")
    dbg(other, "22_other_after_maybe_usa_split")

    # Normalize to lowercase and check substring presence
    def match_local_keywords(df, keywords):
        matched_idx = []
        for i, row in df.iterrows():
            combined = f"{row['OWN1']} {row['OWN2']}".lower()
            if any(kw.lower() in combined for kw in keywords):
                matched_idx.append(i)
        matched_df = df.loc[matched_idx]
        remaining_df = df[~df[KEY].isin(matched_df[KEY])]
        return matched_df, remaining_df
    
    # Keywords to catch local government early
    local_gov_pre_kw = [
        "city of", "town of", "village of",
        "the city of", "the town of", "city", "town", "municipal", "school district"
    ]

    likely_local_gov, other = match_local_keywords(other, local_gov_pre_kw)
    gov = pd.concat([gov, likely_local_gov], ignore_index=True)
    log_totals(gov,   "23_gov_after_early_local")
    dbg(gov, "23_gov_after_early_local")

    log_totals(other, "24_other_after_early_local")
    dbg(other, "24_other_after_early_local")


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
        r'\bU\.?S\.?A?\b',
        r'\bU\s*\.?\s*S\s*\.?\s*A?\s*\.?\b',
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
        r'\bBIA\b',
        r'\bINTR\b',
        r'\bUSDI\b',
        r'\bB\s*L\s*M\b',
    ]))

    government, other = get_corp(other, government_keywords)
    log_totals(government, "25_government_keyword_hits")
    dbg(government, "25_government_keyword_hits")
    log_totals(other,      "26_other_after_government_kw")
    dbg(other, "26_other_after_government_kw")

    # Step 2.1: Filter out corporate-style names again from gov matches
    # corp_like_gov = government['Simple_Owners'].str.contains(corp_filter, regex=True)
    corp_like_gov = (
        government['OWN1'].str.contains(corp_filter, regex=True, case=False, na=False) 
        #| government['OWN2'].str.contains(corp_filter, regex=True, case=False, na=False)
    )

    gov_add = government[~corp_like_gov]
    unk_gov = government[corp_like_gov]

    gov = pd.concat([gov, gov_add]).drop_duplicates(subset=[KEY])

    # Put misclassified corporates back into other
    other = pd.concat([other, unk_gov]).drop_duplicates(subset=[KEY])
    log_totals(gov,   "27_gov_after_corp_like_filter")
    log_totals(other, "28_other_after_reassign_corp_like_gov")

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
    gov = confirmed_gov.drop_duplicates(subset=[KEY])
    other = pd.concat([other, misclassified_corp]).drop_duplicates(subset=[KEY])
    log_totals(gov,   "29_confirmed_gov_final")
    dbg(gov, "29_confirmed_gov_final")
    log_totals(other, "30_other_after_final_corp_like")
    dbg(other, "30_other_after_final_corp_like")

    other_family = other[other['initial_class'] == 1]
    other = other[other['initial_class'] != 1]
    family = pd.concat([other_family, family])
    family['Own_Type'] = 45
    log_totals(family, "31_family_after_other_family")
    dbg(family, "31_family_after_other_family")
    log_totals(other,  "32_other_after_remove_other_family")
    dbg(other, "32_other_after_remove_other_family")

    corp, other = get_corp(other, corp_keywords)
    corp['Own_Type'] = 41
    log_totals(corp,  "33_corp_bucket")
    dbg(corp, "33_corp_bucket")
    log_totals(other, "34_other_for_model")
    dbg(other, "34_other_for_model")

    if is_df_parallel:
        other['Simple_Owners'] = parallelize_on_rows(other[['Simple_Owners']], preprocess_df_simple_owner)
    else:
        other['Simple_Owners'] = other['Simple_Owners'].apply(preprocess_simple_owner)
    log_totals(other, "35_other_after_tokenize")
    dbg(other, "35_other_after_tokenize")

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
    log_totals(other, "36_other_after_model_predictions")
    dbg(other, "36_other_after_model_predictions")

    # other = other[~other.PRCLDMPID.isin(gov.PRCLDMPID)]

    total = pd.concat([
        other, trust43, family, gov, religious_groups,
        c42, c43, null, corp
    ])
    log_totals(total, "37_total_pre_gov_subclassify")
    dbg(total, "37_total_pre_gov_subclassify")

    gov = total[total['Own_Type'] == 0]
    not_gov = total[total['Own_Type'] != 0]
    log_totals(gov,     "38_gov_for_subclassify")
    log_totals(not_gov, "39_not_gov_for_subclassify")

    # Step 1: Classify Federal Government
    federal_kw = federal_kw + [ 
        r'\bU\.?S\.?A?\b',
        r'\bU\s*\.?\s*S\s*\.?\s*A?\s*\.?\b',
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
    fed_gov['Own_Type'] = 25
    log_totals(fed_gov, "40_fed_gov")
    dbg(fed_gov, "40_fed_gov")
    log_totals(remaining_gov, "41_remaining_after_fed")
    dbg(remaining_gov, "41_remaining_after_fed")

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
        r'\bIRRIGATION\b',
        r'\bSEWER\b',
        r'\bDRAINAGE\b',
        r'\bSANITATION\b',
        r'\Board OF (Education)\b'
    ]

    local_pattern = '|'.join(local_keywords)
    local_gov, remaining_gov = get_gov_row(remaining_gov, [local_pattern])
    local_gov['Own_Type'] = 32
    log_totals(local_gov, "42_local_gov")
    dbg(local_gov, "42_local_gov")
    log_totals(remaining_gov, "43_remaining_after_local")
    dbg(remaining_gov, "43_remaining_after_local")

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
    log_totals(state_gov, "44_state_gov")
    dbg(state_gov, "44_state_gov")
    log_totals(remaining_gov, "45_remaining_after_state")
    dbg(remaining_gov, "45_remaining_after_state")


    # Step 4: Any leftovers are likely local
    remaining_gov['Own_Type'] = 32
    log_totals(remaining_gov, "46_remaining_gov_as_local")

    # Step 5: Combine all
    total = pd.concat([fed_gov, local_gov, state_gov, remaining_gov, not_gov])
    log_totals(total, "47_total_pre_dedupe")
    dbg(total, "47_total_pre_dedupe")

    total = total.drop_duplicates(['PRCLDMPID'], keep='first')
    log_totals(total, "48_total_post_dedupe")
    dbg(total, "48_total_post_dedupe")

    # --- FINAL GUARD: start vs end ---
    log_totals(total, "99_before_write", start_ids=_start_ids)
    dbg(total, "99_before_write")
    _end_rows = len(total)
    if _end_rows != _start_rows:
        end_ids = set(total[KEY].dropna().astype(str)) if KEY in total.columns else set()
        missing = _start_ids - end_ids
        extra   = end_ids - _start_ids
        _write_series(os.path.join(DEBUG_DIR, "final_missing_ids_classify.csv"), KEY, missing)
        _write_series(os.path.join(DEBUG_DIR, "final_extra_ids_classify.csv"),   KEY, extra)
        raise RuntimeError(
            f"[classify_unknowns] Row count mismatch: start={_start_rows}, end={_end_rows}. "
            f"Missing={len(missing)}, Extra={len(extra)}. "
            f"See {DEBUG_DIR}/final_missing_ids_classify.csv and final_extra_ids_classify.csv"
        )
    else:
        print(f"[OK] classify_unknowns row count preserved: {_end_rows} rows")
    # --- END FINAL GUARD ---

    print("[diag] Own_Type distribution at end of Classify_Unknowns:")
    print(total['Own_Type'].value_counts(dropna=False))

    gc.collect()
    total.to_csv(output_path)
    print(f'\n✅ Ownerships classified and saved to {output_path}')
