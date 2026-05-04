from __future__ import annotations

import itertools
import re
import unicodedata
import uuid
from typing import Dict

import pandas as pd

from configs import NameCleaner, NamesExpander, biz_word_drop, junior_keywords, keywords

try:
    from metaphone import doublemetaphone
except ImportError as exc:
    raise ImportError(
        "The 'metaphone' package is required for Deduplication_Tool."
    ) from exc


TARGET_NAME_FIELD = "FIRST_LABEL_LINE"
ADDRESS_FIELDS = ["ADDRESS1", "CITY", "STATEAB", "ZIP_CD"]
ADDRESS_DEDUPE_FIELD = "Address_Unq_ID"
GENERIC_SUFFIX_TOKENS = ["JR", "II", "III", "IV", "ESTATE", "FAMILY", "TRUST"]


def create_regex_pattern_from_list(word_list):
    joined = "|".join(v.strip() for v in word_list if v)
    return re.compile(r"\b(?:%s)\b" % joined, flags=re.I)


def generate_normalized_name(name_tuple):
    return "".join(name_tuple).upper()


def normalize_unicode_to_ascii(data):
    val = unicodedata.normalize("NFKD", data).encode("ASCII", "ignore").decode("utf-8")
    val = val.replace("'", "").replace("`", "")
    val = re.sub("[^A-Za-z0-9 ]+", " ", val)
    val = re.sub(" +", " ", val)
    return val.strip()


def normalize_zip_code(value) -> str:
    digits = re.sub(r"\D+", "", normalize_unicode_to_ascii("" if pd.isna(value) else str(value)))
    return digits


def generate_combinations(name_tuple):
    combinations = [tuple(name_tuple)]
    if len(name_tuple) > 2:
        combinations.extend(itertools.combinations(name_tuple, len(name_tuple) - 1))
    return combinations


def canonicalize_owner_tokens(tokens):
    core_tokens = [token for token in tokens if token not in GENERIC_SUFFIX_TOKENS]
    suffix_tokens = [token for token in GENERIC_SUFFIX_TOKENS if token in tokens]
    return core_tokens + suffix_tokens


def build_owner_keys(tokens, initial_class):
    canonical_tokens = canonicalize_owner_tokens(tokens)

    if not canonical_tokens:
        return [tuple()]

    strict_matching = (
        initial_class in (1, 3)
        or any(token in GENERIC_SUFFIX_TOKENS for token in canonical_tokens)
    )

    if strict_matching:
        return [tuple(canonical_tokens)]

    return generate_combinations(canonical_tokens)


def compute_initial_class(owner_series: pd.Series) -> pd.Series:
    init_df = pd.DataFrame(index=owner_series.index)
    own_text = owner_series.fillna("").astype(str)

    wrd_srch = create_regex_pattern_from_list(keywords)
    is_junior = create_regex_pattern_from_list(junior_keywords)

    init_df["len_owner"] = own_text.apply(lambda x: len(x.split()))

    non_std_naming = init_df["len_owner"] == 1
    nan_owner = owner_series.isna() | own_text.str.strip().eq("")
    corporate = own_text.str.contains(wrd_srch.pattern, regex=True, case=False)
    juniors = own_text.str.contains(is_junior.pattern, regex=True, case=False)

    init_df["initial_class"] = 10
    init_df.loc[nan_owner, "initial_class"] = 2
    init_df.loc[(non_std_naming) & (~nan_owner), "initial_class"] = 0
    init_df.loc[init_df["len_owner"] > 1, "initial_class"] = 1
    init_df.loc[juniors, "initial_class"] = 3
    init_df.loc[corporate, "initial_class"] = 0

    return init_df["initial_class"]


def preprocess_names(owner: str | None):
    owner = owner or ""
    owner = owner.replace("'", "").replace("`", "")

    for token in NameCleaner + biz_word_drop:
        owner = re.sub(re.escape(token), " ", owner, flags=re.I)

    for key, value in NamesExpander.items():
        owner = re.sub(key, value, owner)

    owner = re.sub(r"\b[a-zA-Z]\b", "", owner)

    simple_owner = normalize_unicode_to_ascii(owner)
    tokens = [token for token in simple_owner.split(" ") if token]
    return simple_owner, tokens


def normalize_owner_order(owner: str):
    if owner.count(",") != 1:
        return owner

    if re.search(r"\d", owner):
        return owner

    if create_regex_pattern_from_list(biz_word_drop).search(owner):
        return owner

    left, right = [part.strip() for part in owner.split(",", 1)]

    if not left or not right:
        return owner

    return f"{right} {left}".strip()


def build_address_key(row: pd.Series) -> str | None:
    normalized_parts = [
        normalize_unicode_to_ascii("" if pd.isna(row[field]) else str(row[field])).upper()
        for field in ADDRESS_FIELDS[:-1]
    ]
    normalized_parts.append(normalize_zip_code(row["ZIP_CD"]))

    if not any(normalized_parts):
        return None

    return "|".join(normalized_parts)


def validate_input_columns(df: pd.DataFrame):
    required_columns = [TARGET_NAME_FIELD, *ADDRESS_FIELDS]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        joined = ", ".join(missing_columns)
        raise KeyError(f"Input CSV must contain columns: {joined}")


def preprocess_table(df0: pd.DataFrame) -> pd.DataFrame:
    validate_input_columns(df0)

    df = df0.copy()

    df[TARGET_NAME_FIELD] = df[TARGET_NAME_FIELD].replace("B L M", "BLM")
    df[TARGET_NAME_FIELD] = df[TARGET_NAME_FIELD].replace("U S FOREST", "US FOREST SERVICE")
    df["initial_class"] = compute_initial_class(df[TARGET_NAME_FIELD])

    df[TARGET_NAME_FIELD] = df[TARGET_NAME_FIELD].fillna("")
    df[TARGET_NAME_FIELD] = df[TARGET_NAME_FIELD].apply(normalize_owner_order)

    processed = df[TARGET_NAME_FIELD].str.upper().apply(preprocess_names)

    df["Simple_Owners"] = processed.apply(lambda x: x[0])
    token_lists = processed.apply(lambda x: x[1])
    df["Owners"] = [
        build_owner_keys(tokens, initial_class)
        for tokens, initial_class in zip(token_lists, df["initial_class"])
    ]
    df["Address_Key"] = df.apply(build_address_key, axis=1)
    df["place_id"] = range(len(df))
    df["Unq_ID"] = None

    return df


def add_combinations_to_directory(owners):
    meta_list = []
    seen = set()

    for owner_tokens in owners:
        concat_name = generate_normalized_name(tuple(owner_tokens))
        primary_code = doublemetaphone(concat_name)[0]

        if primary_code and primary_code not in seen:
            meta_list.append(primary_code)
            seen.add(primary_code)

    return meta_list


def compute_dict(df: pd.DataFrame) -> Dict[str, list[int]]:
    table_dict: Dict[str, list[int]] = {}

    for key_list, value in df[["Meta_names", "place_id"]].itertuples(index=False):
        for key in key_list:
            table_dict.setdefault(key, []).append(value)

    return table_dict


def matching(meta_list, unqid, table_dict, unqid_dict):
    match_list = [table_dict[m] for m in meta_list if m in table_dict]

    if not match_list:
        return str(uuid.uuid4()) if pd.isna(unqid) else unqid

    result = set(match_list[0])
    for match_set in match_list[1:]:
        result.update(match_set)

    existing_ids = {unqid_dict.get(pid) for pid in result if pid in unqid_dict}
    existing_ids.discard(None)

    uid = sorted(existing_ids)[0] if existing_ids else (str(uuid.uuid4()) if pd.isna(unqid) else unqid)

    for pid in result:
        unqid_dict[pid] = uid

    return uid


def assign_address_ids(table: pd.DataFrame) -> pd.Series:
    address_ids = pd.Series(index=table.index, dtype="object")

    for _, group in table.groupby("Unq_ID", sort=False):
        group_address_ids: dict[str, str] = {}

        for idx, address_key in group["Address_Key"].items():
            if not address_key:
                address_ids.at[idx] = str(uuid.uuid4())
                continue

            if address_key not in group_address_ids:
                group_address_ids[address_key] = str(uuid.uuid4())

            address_ids.at[idx] = group_address_ids[address_key]

    return address_ids


def deduplicate_table(df: pd.DataFrame) -> pd.DataFrame:
    table = preprocess_table(df)
    table["Meta_names"] = table["Owners"].apply(add_combinations_to_directory)

    table_dict = compute_dict(table)
    unqid_dict = {}

    table["Unq_ID"] = [
        matching(meta_list, unqid, table_dict, unqid_dict)
        for meta_list, unqid in table[["Meta_names", "Unq_ID"]].itertuples(index=False)
    ]
    table[ADDRESS_DEDUPE_FIELD] = assign_address_ids(table)
    table["Total_Parcels_Owned"] = table.groupby("Unq_ID")["Unq_ID"].transform("count")

    return table


def deduplicate_csv(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    table = pd.read_csv(input_path, low_memory=False)
    output = deduplicate_table(table)

    if output_path:
        output.to_csv(output_path, index=False)

    return output
