from pathlib import Path
import pandas as pd
import multiprocessing as mp
import os

# ------------------------------------------------------------------
# Parallel settings
# ------------------------------------------------------------------
use_parallel = True
is_df_parallel = True
num_processes = None

# ------------------------------------------------------------------
# Batch settings
# ------------------------------------------------------------------
chunksize = 1000
decoding_batch_size = 1000000
idlc_batch_size = 200000

# ------------------------------------------------------------------
# Resolve project paths (robust)
# ------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", _THIS_DIR)).resolve()

DATA_DIR = PROJECT_DIR / "data"
INPUT_DIR = PROJECT_DIR / "AWS_INPUTS"

# ------------------------------------------------------------------
# Input / Output files
# ------------------------------------------------------------------
output_classify_unknown = "new_parallel_state_temp.csv"

input_name_matching = "new_parallel_state_temp.csv"
output_name_matching = "Full_Data_Table.csv"

input_classified_state = "new_classified_state_temp.csv"
input_id_dictionary = "ID_Dictionary.csv"
input_array = "ID_NLCD_temp.tif"

output_land_analysis = "Full_Data_Table.csv"

input_summary_script = "Full_Data_Table.csv"

# ------------------------------------------------------------------
# CPU configuration
# ------------------------------------------------------------------
if num_processes is None:
    num_processes = max(mp.cpu_count() - 2, 1)

# ------------------------------------------------------------------
# Load keyword data
# ------------------------------------------------------------------
KEYWORDS_PATH = DATA_DIR / "keywords.csv"
keyword_lists = pd.read_csv(KEYWORDS_PATH)

# Drop first row (legacy behavior preserved)
keyword_lists = keyword_lists.drop(keyword_lists.index[0])

trust_kw = list(keyword_lists["trust_kw"].dropna().str.replace("'", ""))
corp_keywords = list(keyword_lists["corp_keywords"].dropna().str.replace("'", ""))
government_keywords = list(keyword_lists["government_keywords"].dropna().str.replace("'", ""))
government_keywords_plus = list(keyword_lists["government_keywords_plus"].dropna().str.replace("'", ""))
rel_key_words = list(keyword_lists["rel_key_words"].dropna().str.replace("'", ""))
ckw = list(keyword_lists["ckw"].dropna().str.replace("'", ""))
kw42 = list(keyword_lists["kw42"].dropna().str.replace("'", ""))
kw43 = list(keyword_lists["kw43"].dropna().str.replace("'", ""))
federal_kw = list(keyword_lists["federal_kw"].dropna().str.replace("'", ""))

# ------------------------------------------------------------------
# Name processing lists
# ------------------------------------------------------------------
keywords = [
    ' BANK ', ' CORP', ' LLC', ' INC', ' LTD', ' HRS', 'MGT',
    'CORPORATION', 'PARTICIPATION', ' TRUST', ' TRUS',
    ' CO ', ' LP', 'UNIVERSITY', 'COLLEGE', ' CHURCH', 'STATE',
    ' CLUB', 'BAPTISI', 'EVANGELICAL', 'METHODIST', 'CATHOLIC',
    'PROPERTIES', 'ASSOCIATIONS', 'ASSOCIATES', ' TOWN OF',
    'CITY OF', ' ASSOCI', 'SOCIETY', 'MAINTENANCE', 'MAINTENANC',
    ' COUNTY', 'TELEPHONE', 'ELECTRIC', 'ENTERPRISES', 'ENTERPRISE',
    'AUTHORITY', 'HOMEOWNERS', 'INTERNATIONAL', ' MINISTRY',
    ' OFFICE', 'INVESTMENT', ' HOME', ' MGMT', ' PRESBYTERIAN',
    ' INN', 'ASSOCIATION', ' STE ', ' L L C', 'ACCOUNTING',
    'MAINTEN', 'PRODUCT', 'MUTUAL', 'ESTATES', 'PARTNER',
    ' & SONS', 'FINANCE', ' TITLE', ' L P', ' FARM', 'WIRELESS',
    'COMMUNICATION', ' SERVICE', 'BAR & GRILL', ' DEPT', 'DEPARTMENT',
    ' CTR', ' LOAN', 'SPECIALTIES', 'BRANDS', ' UNITED', 'CREDIT',
    ' UNION', 'CORPORATE', 'TREASURER', ' ADMIN', 'UTILITIES',
    'COMMERCIAL', ' STORAGE', 'FAMILY', 'AND SONS', "INVESTMENTS", "HOLDINGS", "PROPERTIES", "PARTNERS"
]

junior_keywords = [
    " JR ", " JR. ", " Jr ", " Jr. ", " jr ",
    " jr. ", " JR", " JR.", " Jr", " Jr.", " jr",
    " jr.", ' II ', ' III ', ' IV '
]

NameCleaner = [
    'TTEE', 'DR. ', 'MR. ', 'MS. ', ' MRS. ', 'CAPTAIN', 'CPT.',
    'PROF ', 'REV. COACH ', 'PROFESSOR ', 'REVEREND ', 'SIR ',
    'LT. ', 'SGT. ', 'SR. ', 'Miss'
]

# ------------------------------------------------------------------
# Common name expansions
# ------------------------------------------------------------------
NAME_EXPAND_PATH = DATA_DIR / "Common_Name_Abbreviations.csv"

NamesExpander = pd.read_csv(
    NAME_EXPAND_PATH,
    index_col=0,
    names=["Full"]
).to_dict()["Full"]

# ------------------------------------------------------------------
# Business word drop list
# ------------------------------------------------------------------
biz_word_drop = [
    ' CORP', ' LLC', ' INC', ' LTD', ' HRS', 'MGT', ' OF ', ' CO ',
    ' LP', ' MGMT', ' STE ', ' L L C', ' L P', 'Holdings', 'Partners'
]

# ------------------------------------------------------------------
# NLCD reclassification
# ------------------------------------------------------------------
reclass_dict = {
    0: -99,
    11: 1,
    21: 2,
    22: 3,
    23: 4,
    24: 5,
    31: 6,
    41: 7,
    42: 8,
    43: 9,
    52: 11,
    71: 12,
    81: 13,
    82: 14,
    90: 15,
    95: 16
}

# ------------------------------------------------------------------
# State lookup tables
# ------------------------------------------------------------------
us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

flipped_us_state = {v: k for k, v in us_state_to_abbrev.items()}