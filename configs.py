import pandas as pd
import platform
import multiprocessing as mp
import os

use_parallel = True
is_df_parallel = True
num_processes = None

# Name Matching input
# input_name_matching = 'temp.csv'
# output_name_matching = 'new_parallel_state_temp.csv'

# # Classify Unknown Batch size
# input_classify_unknown = 'new_parallel_state_temp.csv'
# output_classify_unknown = 'new_classified_state_temp.csv'
# chunksize = 1000  # batch size for counts array

# # Classify Unknown Batch size
# input_classify_unknown = 'temp.csv'
output_classify_unknown = 'new_parallel_state_temp.csv'
chunksize = 1000  # batch size for counts array

# # Name Matching input
input_name_matching = 'new_parallel_state_temp.csv'
# output_name_matching = 'new_classified_state_temp.csv'
output_name_matching = 'Full_Data_Table.csv'

# Land Analysis Batch size
input_classified_state = 'new_classified_state_temp.csv'
input_id_dictionary = 'ID_Dictionary.csv'
input_array = 'ID_NLCD_temp.tif'   # or read

output_land_analysis = 'Full_Data_Table.csv'

decoding_batch_size = 1000000  # batching ar array
idlc_batch_size = 200000  # batching idlc dictionary items

# Summmary script
input_summary_script = 'Full_Data_Table.csv'


if num_processes is None:
    num_processes = mp.cpu_count() - 2

project_dir = os.environ.get("PROJECT_DIR")
if project_dir:
    PROJECT_DIR = project_dir
else:
    PROJECT_DIR = os.getcwd()

if platform.system() == 'Windows':
    DATA_DIR = f'{PROJECT_DIR}\\data\\'
else:
    DATA_DIR = f'{PROJECT_DIR}/data/'

INPUT_DIR = f'{PROJECT_DIR}/AWS_INPUTS/'
    
# DATA_DIR = r'D:\Documents\OwnershipMap\New_Script\Test_ENV\INPUTS\\'
# xls = pd.ExcelFile(DATA_DIR+'keywords.xlsx')
# keyword_lists  = pd.read_excel(xls, 'in')
keyword_lists = pd.read_csv(DATA_DIR+'keywords.csv')
keyword_lists = keyword_lists.drop(keyword_lists.index[0])

# keywords = list(keyword_lists['keywords'].dropna().str.replace("'", ""))
# junior_keywords = list(keyword_lists['junior_keywords'].dropna().str.replace('"', ""))
# NameCleaner = list(keyword_lists['NameCleaner'].dropna().str.replace("'", ""))
# biz_word_drop = list(keyword_lists['biz_word_drop'].dropna().str.replace("'", ""))
trust_kw  = list(keyword_lists['trust_kw'].dropna().str.replace("'", ""))
corp_keywords  = list(keyword_lists['corp_keywords'].dropna().str.replace("'", ""))
government_keywords  = list(keyword_lists['government_keywords'].dropna().str.replace("'", ""))
government_keywords_plus  = list(keyword_lists['government_keywords_plus'].dropna().str.replace("'", ""))
rel_key_words  = list(keyword_lists['rel_key_words'].dropna().str.replace("'", ""))
ckw = list(keyword_lists['ckw'].dropna().str.replace("'", ""))
kw42 = list(keyword_lists['kw42'].dropna().str.replace("'", ""))
kw43 = list(keyword_lists['kw43'].dropna().str.replace("'", ""))
federal_kw = list(keyword_lists['federal_kw'].dropna().str.replace("'", ""))
# state_kw = list(keyword_lists['state_kw'].dropna().str.replace("'", ""))


keywords = [' BANK ', ' CORP', ' LLC', ' INC', ' LTD', ' HRS', 'MGT',
            'CORPORATION', 'PARTICIPATION', ' TRUST', ' TRUS', ' OF ',
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
            'COMMERCIAL', ' STORAGE', 'FAMILY', 'AND SONS']

# junior_keywords = ["JR", "JR.", ' II', ' III', ' IV']
junior_keywords = [" JR ", " JR. ", " Jr ", " Jr. ", " jr ",
                   " jr. ", " JR", " JR.", " Jr", " Jr.", " jr",
                   " jr.", ' II ', ' III ', ' IV ']

NameCleaner = ['TTEE', 'DR. ', 'MR. ', 'MS. ', ' MRS. ', 'CAPTAIN', 'CPT.',
               'PROF ', 'REV. COACH ', 'PROFESSOR ', 'REVEREND ', 'SIR ',
               'LT. ', 'SGT. ', 'SR. ', 'Miss']

NamesExpander = pd.read_csv(f'{DATA_DIR}Common_Name_Abbreviations.csv',
                            index_col=0, names=["Full"]).to_dict()['Full']

#Add some 42 and 43 language
#Likely create more specific lists
biz_word_drop = [' CORP', ' LLC', ' INC', ' LTD', ' HRS', 'MGT', ' OF ', ' CO ',
                 ' LP', ' MGMT', ' STE ', ' L L C', ' L P', 'Holdings', 'Partners']

# # Configs for Classify_Unknowns_Application

# # # Trusts keywords
# trust_kw = [' trust ', ' rev tr of ']

# # # corporate keyword in Classify Unknown
# corp_keywords = [' BANK ', 'CORPORATION', 'PARTICIPATION', 'PROPERTIES',
#                  'ASSOCIATIONS', 'ASSOCIATES', ' ASSOCI', 'MAINTENANCE',
#                  'MAINTENANC', 'TELEPHONE', 'ELECTRIC', 'ENTERPRISES',
#                  'ENTERPRISE', 'AUTHORITY', 'HOMEOWNERS', 'INTERNATIONAL',
#                  ' OFFICE', 'INVESTMENT', ' MGMT', ' INN', 'ASSOCIATION',
#                  'ACCOUNTING', 'MAINTEN', 'PRODUCT', 'MUTUAL', ' ESTATES',
#                  ' & SONS', 'FINANCE', ' TITLE', 'WIRELESS', 'COMMUNICATION',
#                  ' SERVICE', 'BAR & GRILL', ' DEPT', ' CTR', ' LOAN',
#                  'SPECIALTIES', 'BRANDS', ' UNITEDCREDIT', ' UNION',
#                  'CORPORATE', 'TREASURER', ' ADMIN', 'UTILITIES',
#                  'COMMERCIAL', ' STORAGE', 'REAL ESTATE', 'CREDIT UNION',
#                  'AND SONS']

# # # government keyword in Classify Unknown
# government_keywords = [' state ', ' county ', ' federal ', ' government ',
#                        ' city of ', ' town of ', ' township ', ' department ',
#                        ' bureau ']

# # # additional check for government keyword in Classify Unknown [gov does not
# # # contain these keywords]
# government_keywords_plus = [' BANK ', ' CORP', 'PARTICIPATION', ' TRUST',
#                             ' TRUS', ' CO ', ' LP ', 'UNIVERSITY', 'COLLEGE',
#                             ' CHURCH', ' CLUB', 'BAPTIST', 'EVANGELICAL',
#                             'METHODIST', 'CATHOLIC', 'PROPERTIES',
#                             'ASSOCIATIONS', 'ASSOCIATES', ' ASSOCI',
#                             'SOCIETY', 'MAINTENANCE', 'MAINTENANC',
#                             'ENTERPRISES', 'ENTERPRISE', 'AUTHORITY',
#                             'HOMEOWNERS', 'INTERNATIONAL', 'INVESTMENT',
#                             ' HOME', ' MGMT', ' PRESBYTERIAN', ' INN',
#                             'ASSOCIATION', ' STE ', 'ACCOUNTING', 'MAINTEN',
#                             'PRODUCT', 'MUTUAL', 'ESTATES', 'PARTNER',
#                             ' & SONS', 'FINANCE', ' TITLE', ' L P ',
#                             ' FARM', 'WIRELESS', 'COMMUNICATION',
#                             ' SERVICE', 'BAR & GRILL', ' CTR ', ' LOAN',
#                             'SPECIALTIES', 'BRANDS', 'CREDIT', 'CORPORATE',
#                             'TREASURER', ' ADMIN', 'UTILITIES', 'COMMERCIAL',
#                             ' STORAGE', 'FAMILY']

# # # Religious keyword in Classify Unknown
# rel_key_words = [' Adventist ', ' Baptist ', ' Brethren ', ' Catholic ',
#                  'Christian Church', ' Church of ', ' Episcopal ', ' Anglican ',
#                  ' Lutheran ', ' Mennonite ', ' Methodist ', ' Moravian ',
#                  'Mormon ', ' Nazarene ', ' Orthodox ', 'Pentecostal ',
#                  ' Presbyterian ', ' Reformed Church ', 'Spiritualist ',
#                  ' Mosques ', ' Islamic Center ', ' Mosque ', ' Masjid ',
#                  ' Synagogues ', ' Temple ', ' Congregation ', ' of bethlehem ',
#                  'kingdom of god', 'christian*church']

# us_state
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
    "South"
    # "FAKE": "FAKE"
    # "NH" : "NH",
    # "CH" : "CH",
    # "SP" : "SP", 
    # "WC" : "WC"
}

flipped_us_state = {us_state_to_abbrev[k]: k for k in us_state_to_abbrev}
