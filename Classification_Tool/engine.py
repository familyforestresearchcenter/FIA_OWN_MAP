from __future__ import annotations

from ast import pattern
from pathlib import Path
from typing import Optional, List, Dict, Any

import re
import unicodedata
import itertools


import nltk
from nltk.stem import PorterStemmer

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import lil_matrix
import numpy as np

from configs import *

from utils.helpers import *

# ------------------------------------------------------------------------------
# Model paths (authoritative for this module)
# ------------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = _THIS_DIR / "classify_unknown_ownership_model.pkl"
VOCAB_PATH = _THIS_DIR / "model_dict.pkl"

# ------------------------------------------------------------------------------
# NLTK resources (mirror legacy behavior)
# ------------------------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

class Trace:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events: list[dict] = []

    def add(self, stage: str, event: str, **fields):
        if not self.enabled:
            return
        rec = {"stage": stage, "event": event}
        rec.update(fields)
        self.events.append(rec)

def chunks_generator(in_counts, n_keys, x_idx, y_idx, chunk_size):
    size_counts = in_counts.shape[0]
    rows = chunk_size
    cols = n_keys
    for pos in range(0, size_counts - chunk_size, chunk_size):
        curr_counts = in_counts[pos:pos + chunk_size, y_idx]
        model_sparse = lil_matrix((rows, cols))
        model_sparse[:, x_idx] = curr_counts
        yield model_sparse


def get_corp(simple_owner: str | None, wrd_list):
    text = simple_owner or ""

    pattern = create_regex_pattern_from_list(wrd_list)

    m = re.search(pattern, text)
    if m:
        return True, m.group(0)

    return False, None


def get_gov_row(own1: str | None, own2: str | None, wrd_list):
    own1 = own1 or ""
    own2 = own2 or ""

    for kw in wrd_list:
        pattern = re.compile(kw, flags=re.IGNORECASE)

        if re.search(pattern, own1) or re.search(pattern, own2):
            return True, kw  # matched, keyword that triggered

    return False, None  # unmatched


def preprocess_simple_owner(text: str | None) -> str:
    if not text:
        return ""

    # mirror legacy: literal replace, NOT regex
    text = text.lower().replace(r"[^\w\s]", "")
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(tok) for tok in nltk.word_tokenize(text))


def filEmptyStringsInOwners(simple_owner: str | None, own1: str | None):
    if simple_owner is None or simple_owner == "":
        return own1 if own1 not in (None, "") else simple_owner
    return simple_owner

def iso_biz(own1: str | None, own2: str | None):
    own1 = own1 or ""
    own2 = own2 or ""

    wrd_srch = create_regex_pattern_from_list(keywords)

    own1_match = bool(re.search(wrd_srch, own1, flags=re.IGNORECASE))
    own2_match = bool(re.search(wrd_srch, own2, flags=re.IGNORECASE))

    part1 = own1 if own1_match else ""
    part2 = own2 if own2_match else ""

    return f"{part1} {part2}".upper().strip()

def compute_initial_class(own1: str | None, own2: str | None):

    own1_val = own1 or ""
    own2_val = own2 or ""

    wrd_srch = create_regex_pattern_from_list(keywords + ckw)
    is_junior = create_regex_pattern_from_list(junior_keywords)

    len_own1 = len(own1_val.split())

    non_std_naming = len_own1 == 1
    nan_own1 = own1 is None
    nan_own2 = own2 is None

    corporate = (
        contains_keyword(own1_val, keywords + ckw) or
        contains_keyword(own2_val, keywords + ckw)
    )

    juniors = (
        is_junior.search(own1_val) is not None or
        is_junior.search(own2_val) is not None
    )

    initial_class = 10

    if nan_own1 and not nan_own2:
        initial_class = -99999

    if nan_own1 and nan_own2:
        initial_class = 2

    if non_std_naming and not nan_own2:
        initial_class = 1

    if non_std_naming and nan_own2:
        initial_class = 0

    if len_own1 > 1:
        initial_class = 1

    if juniors:
        initial_class = 3

    if corporate:
        initial_class = 0

    return initial_class

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


def preprocess_names(owner: str | None):
    owner = owner or ""

    # legacy: regex replace using pattern derived from list
    pattern = create_regex_pattern_from_list(NameCleaner + biz_word_drop)
    owner = re.sub(pattern, "", owner)

    # legacy: dict-based regex expansion
    for k, v in create_regex_pattern_from_dict(NamesExpander).items():
        owner = re.sub(k, v, owner)

    # legacy: remove single-letter tokens
    owner = re.sub(r"\b[a-zA-Z]\b", "", owner)

    simple_owner = normalize_unicode_to_ascii(owner)
    tokens = [t for t in simple_owner.split(" ") if t]
    return simple_owner, tokens


def preprocess_table(own1: str | None, own2: str | None, is_other: bool = False):

    own1_val = own1 or ""
    own2_val = own2 or ""

    if is_other:
        business_name = iso_biz(own1_val, own2_val)
        simple_owner, tokens = preprocess_names(business_name)

        owners = [tuple(tokens)]

        return {
            "Simple_Owners": simple_owner,
            "Owners": owners,
        }

    combined = f"{own1_val} {own2_val}".upper()

    simple_owner, tokens = preprocess_names(combined)

    owners = generate_combinations(tokens)

    return {
        "Simple_Owners": simple_owner,
        "Owners": owners,
    }

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

def match_local_keywords(own1: str | None, own2: str | None, keywords):
    combined = f"{own1 or ''} {own2 or ''}".lower()

    for kw in keywords:
        if kw.lower() in combined:
            return True, kw

    return False, None

def _log_check(trace: Trace | None, stage: str, rule: str, value: Any, result: bool):
    if trace:
        trace.add(
            stage,
            "check",
            rule=rule,
            value=value,
            result=result,
        )


def classify_owner(
    own1: str,
    own2: str | None = None,
    state_name: str | None = None,
    trace: Trace | None = None,
):
    own1 = own1 or ""
    own2 = own2 or ""

    if trace is None:
        trace = Trace(False)

    trace.add("00_loaded", "state", own1=own1, own2=own2)

    # legacy preprocessing replacements
    own1 = own1.replace("B L M", "BLM")
    own1 = own1.replace("U S FOREST", "US FOREST SERVICE")

    prep = preprocess_table(own1, own2)

    simple_owner = prep["Simple_Owners"]
    owners = prep["Owners"]

    initial_class = compute_initial_class(own1, own2)

    if initial_class == -99999:
        initial_class = 1

    Own_Type = None

    simple_owner = filEmptyStringsInOwners(simple_owner, own1)

    trace.add(
        "01_after_simple_owners_fill",
        "state",
        simple_owner=simple_owner,
        initial_class=initial_class,
    )

    unavailable_keywords = [
        "NOT AVAILABLE FROM THE COUNTY",
        "AVAILABLE, NOT",
        "NOT AVAILABLE",
    ]

    is_unavailable = (
        own1 in unavailable_keywords or
        own2 in unavailable_keywords
    )

    trace.add(
        "02_not_available_bucket",
        "check",
        rule="unavailable_keywords",
        own1=own1,
        own2=own2,
        result=is_unavailable,
    )

    if initial_class == 2 or is_unavailable:
        Own_Type = -99

        trace.add(
            "03_null_bucket",
            "assign",
            own_type=-99,
            reason="initial_class==2 or unavailable_keywords",
        )

        return Own_Type, trace

    trace.add(
        "04_after_remove_null_unavailable",
        "state",
        initial_class=initial_class,
    )

    # split to family / other buckets
    if initial_class == 3:
        bucket = "family"
    elif initial_class in (0, 1):
        bucket = "other"
    else:
        bucket = "unknown"

    trace.add(
        "05_family_other_split",
        "state",
        bucket=bucket,
    )

    # family rows terminate immediately in legacy logic
    if bucket == "family":
        Own_Type = 45
        trace.add(
            "05_family_direct",
            "assign",
            own_type=45,
            reason="initial_class_family_bucket",
        )
        return Own_Type, trace

    # -------------------------
    # Trust detection
    # -------------------------

    if bucket == "other":

        is_trust = re.search(
            r"\b(TRUST|REV\s+TR\s+OF)\b",
            simple_owner
        ) is not None

        trace.add(
            "07_trust_check",
            "check",
            rule="trust_keywords",
            simple_owner=simple_owner,
            result=is_trust,
        )

        if is_trust:

            is_family_trust, kw = get_corp(simple_owner, trust_kw)

            trace.add(
                "08_family_trust_check",
                "check",
                rule="trust_kw",
                result=is_family_trust,
            )

            if is_family_trust:
                Own_Type = 45

                trace.add(
                    "09_family_trust_bucket",
                    "assign",
                    own_type=45,
                    reason="family_trust",
                )

                return Own_Type, trace

            else:
                is_43_trust, kw = get_corp(simple_owner, kw43)

                trace.add(
                    "10_trust43_check",
                    "check",
                    rule="kw43",
                    result=is_43_trust,
                )

                if is_43_trust:
                    Own_Type = 43

                    trace.add(
                        "11_trust43_bucket",
                        "assign",
                        own_type=43,
                        reason="trust_kw43",
                    )

                    return Own_Type, trace


        # -------------------------
        # Farm detection
        # -------------------------

        farm_terms = ["farm", "farms"]
        is_farm, kw = get_corp(simple_owner, farm_terms)

        trace.add(
            "12_farm_check",
            "check",
            rule="farm",
            simple_owner=simple_owner,
            result=is_farm,
        )

        if is_farm:

            is_family_farm, kw = get_corp(
                simple_owner,
                [" family ", " brother ", " son ", " daughter "],
            )

            trace.add(
                "13_family_farm_check",
                "check",
                rule="family_farm_keywords",
                result=is_family_farm,
            )

            if is_family_farm:
                Own_Type = 45

                trace.add(
                    "14_family_after_farms",
                    "assign",
                    own_type=45,
                    reason="family_farm",
                )

                return Own_Type, trace

            else:
                trace.add(
                    "15_other_after_farms",
                    "state",
                    bucket="other",
                )


        # -------------------------
        # 42 / religious / 43 corp
        # ------------------------

        kw42_local = [
            # --- Existing canonical orgs ---
            r'\bNATURE CONSERVANCY\b',
            r'\bNATURE\b',
            r'\bCONSERVANCY\b',
            r'\bWILD LANDS?\b',
            r'\bLAND TRUST\b',
            r'\bAUDUBON\b', #I may want to remove this, it causes a ton of false hits
            r'\bWATERSHED TRUST\b',
            r'\bHERITAGE TRUST\b',
            r'\bTRUST FOR PUBLIC LAND\b',
            r'\bWILDLIFE SANCTUARY\b',
            r'\bNATURE CENTER\b',
            r'\bLand Preservation\b',
            r'\b(WILDLIFE|CONSERVATION|HABITAT) (COUNCIL|GROUP|ALLIANCE)\b',
        ]

        is_42, kw = get_corp(simple_owner, kw42_local)

        trace.add(
            "16_c42_check",
            "check",
            rule="kw42",
            simple_owner=simple_owner,
            result=is_42,
        )

        if is_42:
            Own_Type = 42

            trace.add(
                "16_c42_bucket",
                "assign",
                own_type=42,
                reason="kw42_match",
            )

            return Own_Type, trace



        rel_key_words_local = rel_key_words + [
            r'\bCHURCH OF\b',
            r'\bOF .* CHURCH\b',
            r'\bCHURCH AT\b',
            r'\bCHURCH IN\b',
            r'\bFELLOWSHIP CHURCH\b',
            r'\bCHRISTIAN FELLOWSHIP\b',
            r'\bMINISTR(Y|IES)\b',
            r'\bOF GOD\b',
            r'\bKINGDOM OF GOD\b',
            r'\bBIBLE CHURCH\b',
            r'\bMISSIONARY CHURCH\b',
            r'\bCHAPEL\b',
        ]

        is_religious_simple, _ = get_corp(simple_owner, rel_key_words_local)
        is_religious_raw, _ = get_corp(own1 or "", rel_key_words_local)

        is_religious = is_religious_simple or is_religious_raw

        trace.add(
            "17_religious_groups_check",
            "check",
            rule="rel_key_words",
            result=is_religious,
            simple_match=is_religious_simple,
            raw_match=is_religious_raw,
        )

        if is_religious:
            Own_Type = 43

            trace.add(
                "17_religious_groups_bucket",
                "assign",
                own_type=43,
                reason="religious_group",
            )

            return Own_Type, trace


        kw43_local = [
            r'\bLANDOWNER(S)? ASSOCIATION\b',
            r'\bHOMEOWNER(S)? ASSOCIATION\b',
            r'\bSPORTSMAN(S)? CLUB\b',
            r'\bASSOCIATION\b',
            r'\bCLUB\b',
            r'\bFRIENDS OF\b',
            r'\bYMCA\b',
            r'\bYWCA\b',
            r'\bBOYS SCOUTS\b',
            r'\bGIRLS SCOUTS\b',
            r'\bSCOUTS OF AMERICA\b',
            r'\bSCOUTS\b',
            r'\bROD.?GUN\b',
            r'\bGAME CLUB\b',
            r'\bHUNTING CLUB\b',
            r'\bFISHING CLUB\b',
            r'\bHUNTING ASSOCIATION\b',
            r'\bANGLING ASSOCIATION\b',
            r'\bBEAGLING\b',
            r'\bMOTORCYCLE CLUB\b',
            r'\bMOTORCYCLE ASSOC\b',
            r'\bCLUB\b',
            r'\bLANDOWNERS ASSOC\b',
            r'\bHOMEOWNER(S)? CLUB\b',
            r'\bHOMEOWNER(S)? ASSOC\b',
            r'\bHUNT CLUB\b',
            r'\bHUNT ASSOC\b',
            r'\bCAMPING CLUB\b',
            r'\bSPORTSMAN(S)? ASSOC\b',
            r'\bBEAGLE CLUB\b',
            r'\bCONDO ASSOC\b',
            r'\bCOMMUNITY ASSOC\b',
            r'\bBLOOD CENTER\b',
            r'\bBLOOD CTRS?\b',
            r'\bHOMEOWNERS\b',
            r'\bHOUSING AUTHORITY\b'
        ]

        is_43_simple, _ = get_corp(simple_owner, kw43_local)
        is_43_raw, _ = get_corp(own1 or "", kw43_local)

        is_43 = is_43_simple or is_43_raw

        trace.add(
            "18_c43_check",
            "check",
            rule="kw43",
            result=is_43,
        )

        if is_43:

            exclusion_keywords = [r"\bGOLF\b", r"\bWORLDMARK\b"]
            exclusion_pattern = "|".join(exclusion_keywords)

            excluded = (
                re.search(exclusion_pattern, own1 or "", flags=re.IGNORECASE) is not None or
                re.search(exclusion_pattern, own2 or "", flags=re.IGNORECASE) is not None
            )

            trace.add(
                "18_c43_exclusion_check",
                "check",
                rule="kw43_exclusions",
                result=excluded,
            )

            if not excluded:
                Own_Type = 43

                trace.add(
                    "18_c43_bucket",
                    "assign",
                    own_type=43,
                    reason="kw43_match",
                )

                return Own_Type, trace


        # -------------------------
        # Step 1: Identify USA variations
        # -------------------------

        usa_variations = r"\b(U(\s*\.?\s*)S(\s*\.?\s*)A(\s*\.?\s*)?)\b"

        maybe_usa = (
            re.search(usa_variations, own1) is not None or
            re.search(usa_variations, own2) is not None
        )

        trace.add(
            "20_maybe_usa",
            "check",
            rule="usa_variations",
            own1=own1,
            own2=own2,
            result=maybe_usa,
        )

        corp_acronyms = ["LLC", "INC", "CORP", "CO", "LTD", "LP", "LLP", "PLC"]

        corp_filter = (
            create_regex_pattern_from_literals(
                corp_keywords
                + ckw
                + [
                    "COMPANY",
                    "INSURANCE",
                    "BANK",
                    "MORTGAGE",
                    "SAVINGS",
                    "FINANCIAL",
                    "ASSOCIATION",
                    "COOPERATIVE",
                    "MERGENTHALER",
                    "HOUSING AUTHORITY",
                    "AFRAME PIPE",
                ]
            )
            + "|"
            + create_regex_pattern_from_raw(acronym_regex_variants(corp_acronyms))
        )

        trace.add(
            "21_build_corp_filter",
            "state",
            rule="corp_filter_regex_constructed",
        )
        
        # -------------------------
        # Step 1.1: Filter corporate-style USA names
        # -------------------------

        corp_like_usa = False

        if maybe_usa:
            corp_like_usa = (
                re.search(corp_filter, own1) is not None or
                re.search(corp_filter, own2) is not None
            )

        trace.add(
            "22_corp_like_usa_check",
            "check",
            rule="corp_filter",
            own1=own1,
            own2=own2,
            result=corp_like_usa,
        )

        if maybe_usa and not corp_like_usa:
            bucket = "gov"

            trace.add(
                "23_gov_bucket",
                "state",
                reason="usa_without_corp_pattern",
            )

        elif maybe_usa and corp_like_usa:
            bucket = "usa_corp"

            trace.add(
                "24_usa_corp_bucket",
                "state",
                reason="usa_with_corp_pattern",
            )

        # -------------------------
        # Move records after USA split
        # -------------------------

        if maybe_usa and not corp_like_usa:
            bucket = "gov"

            trace.add(
                "21_gov_from_maybe_usa",
                "state",
                bucket="gov",
                reason="usa_without_corp_pattern",
            )

        elif maybe_usa and corp_like_usa:
            bucket = "other"

            trace.add(
                "22_other_after_maybe_usa_split",
                "state",
                bucket="other",
                reason="usa_corp_pattern",
            )


        # -------------------------
        # Early local government detection
        # -------------------------

        local_gov_pre_kw = [
            "city of",
            "town of",
            "village of",
            "the city of",
            "the town of",
            "city",
            "town",
            "municipal",
            "school district",
        ]

        matched, kw = match_local_keywords(own1, own2, local_gov_pre_kw)

        trace.add(
            "23_early_local_gov_check",
            "check",
            rule="local_gov_pre_kw",
            keyword=kw,
            result=matched,
        )

        if matched:
            bucket = "gov"

            trace.add(
                "23_gov_after_early_local",
                "state",
                bucket="gov",
                reason="early_local_keyword",
            )
        else:
            trace.add(
                "24_other_after_early_local",
                "state",
                bucket="other",
            )

        # -------------------------
        # Step 2: Build government keyword set
        # -------------------------

        government_keywords_full = list(
            set(
                government_keywords
                + [
                    r"\bUNIVERSITY\b",
                    r"\bUNIVERSITY OF\b",
                    r"\bSTATE UNIVERSITY\b",
                    r"\bPUBLIC UNIVERSITY\b",
                    r"\bSTATE COLLEGE\b",
                    r"\bCOLLEGE OF\b",
                    r"\bCOMMUNITY COLLEGE\b",
                    r"\bU\.?S\.?A?\b",
                    r"\bU\s*\.?\s*S\s*\.?\s*A?\s*\.?\b",
                    r"\bFEDERAL\b",
                    r"\bCONSERVATION\b",
                    r"\bGOVT\b",
                    r"\bDEPARTMENT OF (AGRICULTURE|INTERIOR|DEFENSE|ENERGY|EDUCATION|TRANSPORTATION|JUSTICE|LABOR|COMMERCE)\b",
                    r"\bBUREAU OF\b",
                    r"\bUSDA\b",
                    r"\bFOREST SERVICE\b",
                    r"\bEPA\b",
                    r"\bDHS\b",
                    r"\bFBI\b",
                    r"\bDOI\b",
                    r"\bUSFS\b",
                    r"\bFWS\b",
                    r"\bUSFWS\b",
                    r"\bDOT\b",
                    r"\bUSDI\b",
                    r"\bUSACE\b",
                    r"\bNOAA\b",
                    r"\bNPS\b",
                    r"\bDOD\b",
                    r"\bBLM\b",
                    r"\bDOE\b",
                    r"\bBIA\b",
                    r"\bINTR\b",
                    r"\bUSDI\b",
                    r"\bB\s*L\s*M\b",
                    r"\bREGIONAL COUNCIL\b",          
                    r"\bCOUNCIL OF GOVERNMENTS?\b",   
                    r"\bREGIONAL PLANNING (COMMISSION|COUNCIL)\b",
                    r"\bPLANNING COMMISSION\b",
                    r"\bAREA AGENCY ON AGING\b",      
                    r"\bMETROPOLITAN PLANNING ORGANIZATION\b",
                    r"\bMPO\b",
                    r"\bREGIONAL DISTRICT\b",
                    r"\bWATER DISTRICT\b",
                    r"\bUTILITY DISTRICT\b",
                ]
            )
        )

        trace.add(
            "25_build_gov_keyword_set",
            "state",
            keyword_count=len(government_keywords),
        )

        # -------------------------
        # Step 2: Government keyword match
        # -------------------------

        is_government, kw = get_gov_row(own1, own2, government_keywords_full)

        trace.add(
            "26_government_keyword_check",
            "check",
            rule="government_keywords",
            simple_owner=simple_owner,
            result=is_government,
        )

        if is_government:
            bucket = "gov"

            trace.add(
                "27_government_keyword_hits",
                "state",
                bucket="gov",
                reason="government_keyword_match",
            )
        else:
            trace.add(
                "28_other_after_government_kw",
                "state",
                bucket="other",
            )

        # -------------------------
        # Step 2.1: Filter corporate-style gov names
        # -------------------------

        # Tier 1 (legal) + Tier 2 (strong business identity)
        corp_override_patterns = [
            # --- Tier 1 ---
            r"\bLLC\b", r"\bL\.?L\.?C\.?\b",
            r"\bINC\b", r"\bINCORPORATED\b",
            r"\bCORP\b", r"\bCORPORATION\b",
            r"\bLTD\b", r"\bLIMITED\b",
            r"\bLP\b", r"\bL\.?P\.?\b",
            r"\bLLP\b", r"\bPLC\b",
            r"\bPC\b",

            # --- Tier 2 ---
            r"\bCOMPANY\b", r"\bCO\b",
            r"\bHOLDINGS?\b",
            r"\bINVESTMENTS?\b",
            r"\bENTERPRISES?\b",
            r"\bPROPERTIES\b",
            r"\bREALTY\b",
            r"\bCAPITAL\b",
            r"\bVENTURES?\b",
            r"\bDEVELOPMENT\b",
        ]

        corp_override_regex = re.compile("|".join(corp_override_patterns), re.IGNORECASE)

        corp_like_gov = False

        if is_government:
            corp_like_gov = corp_override_regex.search(own1) is not None

        trace.add(
            "29_corp_like_gov_check",
            "check",
            rule="corp_override_patterns",
            simple_owner=simple_owner,
            result=corp_like_gov,
        )

        if is_government and not corp_like_gov:
            bucket = "gov"

            trace.add(
                "30_gov_after_corp_filter",
                "state",
                bucket="gov",
                reason="government_keyword_without_corp_pattern",
            )

        elif is_government and corp_like_gov:
            bucket = "unk_gov"

            trace.add(
                "31_unknown_gov_bucket",
                "state",
                bucket="unk_gov",
                reason="government_keyword_with_corp_pattern",
            )

        # -------------------------
        # Reassign corp-like gov matches back to other
        # -------------------------

        if is_government and not corp_like_gov:
            bucket = "gov"

            trace.add(
                "32_gov_after_corp_like_filter",
                "state",
                bucket="gov",
                reason="government_keyword_without_corp_pattern",
            )

        elif is_government and corp_like_gov:
            bucket = "other"

            trace.add(
                "33_other_after_reassign_corp_like_gov",
                "state",
                bucket="other",
                reason="corp_like_removed_from_gov",
            )


        # -------------------------
        # Final corp-like gov safety check
        # -------------------------

        corp_like_gov_final = corp_override_regex.search(own1) is not None

        trace.add(
            "34_corp_like_gov_final_check",
            "check",
            rule="corp_filter_final",
            own1=own1,
            result=corp_like_gov_final,
        )


        # -------------------------
        # Split confirmed gov vs misclassified corporate
        # -------------------------

        if is_government and not corp_like_gov_final:
            confirmed_gov = True
            misclassified_corp = False
        else:
            confirmed_gov = False
            misclassified_corp = True

        trace.add(
            "35_gov_split",
            "state",
            confirmed_gov=confirmed_gov,
            misclassified_corp=misclassified_corp,
        )

        # -------------------------
        # Assign corporate (41) if not government
        # -------------------------

        if not confirmed_gov and corp_like_gov_final:

            trace.add(
                "36_corp_assignment",
                "assign",
                own_type=41,
                reason="corp_filter_final_match",
            )

            return 41, trace

        # -------------------------
        # Assign government class
        # -------------------------

        if confirmed_gov:
            Own_Type = 0

            trace.add(
                "37_assign_gov",
                "assign",
                own_type=0,
                reason="confirmed_government",
            )

        # -------------------------
        # Reassign corp-like gov rows back to other
        # -------------------------

        if confirmed_gov:
            trace.add(
                "38_confirmed_gov_final",
                "state",
                bucket="gov",
            )

        elif misclassified_corp:
            bucket = "other"

            trace.add(
                "39_other_after_final_corp_like",
                "state",
                bucket="other",
                reason="corp_like_removed_from_gov_final",
            )

        
        if Own_Type != 0:

            # -------------------------
            # Move other-family rows to family bucket
            # -------------------------

            if initial_class == 1:
                Own_Type = 45

                trace.add(
                    "40_family_after_other_family",
                    "assign",
                    own_type=45,
                    reason="initial_class_family",
                )

                return Own_Type, trace

            trace.add(
                "41_other_after_remove_other_family",
                "state",
                bucket="other",
            )

            # -------------------------
            # Corporate classification
            # -------------------------

            is_corp, kw = get_corp(simple_owner, corp_keywords)

            trace.add(
                "42_corp_check",
                "check",
                rule="corp_keywords",
                simple_owner=simple_owner,
                result=is_corp,
            )

            if is_corp:
                Own_Type = 41

                trace.add(
                    "43_corp_bucket",
                    "assign",
                    own_type=41,
                    reason="corp_keyword_match",
                )

                return Own_Type, trace

            trace.add(
                "44_other_for_model",
                "state",
                bucket="other",
            )

            # -------------------------
            # Random forest fallback
            # -------------------------

            # replicate legacy tokenization
            tokenized_owner = preprocess_simple_owner(simple_owner)

            trace.add(
                "45_other_after_tokenize",
                "state",
                tokenized_owner=tokenized_owner,
            )

            Own_Type = None

            if tokenized_owner and tokenized_owner.strip():

                try:
                    # load trained artifacts FIRST
                    model = load_model(VOCAB_PATH)          # dict vocab
                    classify_model = load_model(MODEL_PATH) # RF model

                    model_key = np.sort(list(model.keys()))

                    # IMPORTANT: use fixed vocab
                    vectorizer = TfidfVectorizer(vocabulary=model)

                    # use transform, not fit_transform
                    counts = vectorizer.transform([tokenized_owner])
                    counts_key = vectorizer.get_feature_names_out()

                    # align feature spaces exactly like legacy
                    _, x_ind, y_ind = np.intersect1d(
                        model_key,
                        counts_key,
                        return_indices=True
                    )

                    # construct sparse vector in model feature space
                    model_sparse = lil_matrix((1, model_key.shape[0]))

                    if len(x_ind) > 0:
                        model_sparse[:, x_ind] = counts[:, y_ind]

                        prediction = classify_model.predict(model_sparse)[0]
                        Own_Type = int(prediction)

                        trace.add(
                            "46_other_after_model_predictions",
                            "assign",
                            own_type=Own_Type,
                            reason="random_forest_model",
                        )
                    else:
                        trace.add(
                            "46_rf_no_vocab_overlap",
                            "state",
                            reason="no_shared_tokens_with_model_vocab",
                        )

                except Exception as e:
                    trace.add(
                        "46_rf_error",
                        "state",
                        error=str(e),
                    )

            else:
                trace.add(
                    "45_rf_skipped_empty",
                    "state",
                    reason="empty_tokenized_owner",
                )

            # -------------------------
            # HARD FALLBACK (UNKNOWN)
            # -------------------------

            if Own_Type is None:
                Own_Type = -99  # explicit unknown

                trace.add(
                    "46_rf_fallback_unknown",
                    "assign",
                    own_type=Own_Type,
                    reason="rf_failed_unknown",
                )

    # -------------------------
    # Government subclassification entry
    # -------------------------

    trace.add(
        "47_pre_gov_subclassify",
        "state",
        own_type=Own_Type,
    )

    if Own_Type != 0:
        return Own_Type, trace

    # -------------------------
    # Step 1: Federal government
    # -------------------------

    federal_patterns = federal_kw + [
        r"\bU\.?S\.?A?\b",
        r"\bU\s*\.?\s*S\s*\.?\s*A?\s*\.?\b",
        r"\bFEDERAL\b",
        r"\bGOVT\b",
        r"\bDEPARTMENT OF (AGRICULTURE|INTERIOR|DEFENSE|ENERGY|EDUCATION|JUSTICE|LABOR|COMMERCE)\b",
        r"\bBUREAU OF\b",
        r"\bUSDA\b",
        r"\bFOREST SERVICE\b",
        r"\bEPA\b",
        r"\bDHS\b",
        r"\bFBI\b",
        r"\bDOI\b",
        r"\bUSFS\b",
        r"\bFWS\b",
        r"\bUSFWS\b",
        r"\bUSDI\b",
        r"\bUSACE\b",
        r"\bNOAA\b",
        r"\bNPS\b",
        r"\bDOD\b",
        r"\bBLM\b",
        r"\bDOE\b",
        r"\bBIA\b",
        r"\bINTR\b",
        r"\bUSDI\b",
        r"\bB\s*L\s*M\b",
        r"\bAmerica\b",
    ]

    fed_match, kw = get_gov_row(own1, own2, federal_patterns)

    trace.add(
        "48_fed_gov_check",
        "check",
        rule="federal_keywords",
        keyword=kw,
        result=fed_match,
    )

    if fed_match:
        Own_Type = 25

        trace.add(
            "49_fed_gov",
            "assign",
            own_type=25,
            reason="federal_keyword_match",
        )

        return Own_Type, trace

    # -------------------------
    # Step 2: Local government
    # -------------------------

    local_keywords = [
        r"\bCITY\b",
        r"\bTOWN\b",
        r"\bVILLAGE\b",
        r"\bCOUNTY\b",
        r"\bPARISH\b",
        r"\bBOROUGH\b",
        r"\bCOMMUNITY COLLEGE\b",
        r"\bMUNICIPAL\b",
        r"\bSCHOOL DISTRICT\b",
        r"\bFIRE DISTRICT\b",
        r"\bPOLICE DEPARTMENT\b",
        r"\bIRRIGATION\b",
        r"\bSEWER\b",
        r"\bDRAINAGE\b",
        r"\bSANITATION\b",
        r"\bBOARD OF (EDUCATION)\b",
        r"\bWATER RESOURCES\b",
        r"\bWATER DEPARTMENT\b",
        r"\bPUBLIC WORKS\b",
        r"\bUTILITIES\b",
        r"\bUTILITY\b",
        r"\bSANITATION\b",
        r"\bSEWER\b",
        r"\bDRAINAGE\b",
        r"\bIRRIGATION\b",
        r"\bWATER AUTHORITY\b",
        r"\bREGIONAL COUNCIL\b",          
        r"\bCOUNCIL OF GOVERNMENTS?\b",   
        r"\bREGIONAL PLANNING (COMMISSION|COUNCIL)\b",
        r"\bPLANNING COMMISSION\b",
        r"\bAREA AGENCY ON AGING\b",      
        r"\bMETROPOLITAN PLANNING ORGANIZATION\b",
        r"\bMPO\b",
        r"\bREGIONAL DISTRICT\b",
        r"\bWATER DISTRICT\b",
        r"\bUTILITY DISTRICT\b",
        r"\bSANITATION DISTRICT\b",
        r"\bIRRIGATION DISTRICT\b",
        r"\bFIRE DISTRICT\b",
    ]

    local_match, kw = get_gov_row(own1, own2, local_keywords)

    trace.add(
        "50_local_gov_check",
        "check",
        rule="local_keywords",
        keyword=kw,
        result=local_match,
    )

    if local_match:
        Own_Type = 32

        trace.add(
            "51_local_gov",
            "assign",
            own_type=32,
            reason="local_keyword_match",
        )

        return Own_Type, trace

    # -------------------------
    # Step 3: State government
    # -------------------------

    state_name = (state_name or "").upper()

    state_keywords = [
        r"\bSTATE\b.*\b(DEPARTMENT|DEPT|UNIVERSITY|COLLEGE|OFFICE|AGENCY|AUTHORITY|SCHOOL|EDUCATION|COMMISSION)\b",
        r"\bCOMMONWEALTH\b",
        r"\bSTATE OF\b",
        r"\bSTATE OF \w+\b.*\b(DEPARTMENT|DEPT)\b",
        r"\bSTATE \w+ DEPT\b",
        r"\bDEPARTMENT\b",
        r"\bDEPT\b",
        r"\bSTATE\b",
        r"\bDOT\b",
        r"\bDEPARTMENT OF (TRANSPORTATION)\b",
    ]

    # If state context exists (legacy behavior)
    if state_name:
        state_keywords.append(flipped_us_state[state_name].upper())
        state_keywords.append(state_name)

    # If no state context (single-string classifier)
    else:
        state_keywords.extend([rf"\b{s.upper()}\b" for s in us_state_to_abbrev.keys()])
        state_keywords.extend([rf"\b{abbr}\b" for abbr in us_state_to_abbrev.values()])


    state_match, kw = get_gov_row(own1, own2, state_keywords)

    trace.add(
        "52_state_gov_check",
        "check",
        rule="state_keywords",
        keyword=kw,
        result=state_match,
    )

    if state_match:
        Own_Type = 31

        trace.add(
            "52_state_gov",
            "assign",
            own_type=31,
            reason="state_keyword_match",
        )

        return Own_Type, trace

    # -------------------------
    # Step 4: Remaining gov → local
    # -------------------------

    Own_Type = 32

    trace.add(
        "53_remaining_gov_as_local",
        "assign",
        own_type=32,
        reason="gov_fallback_local",
    )

    return Own_Type, trace
