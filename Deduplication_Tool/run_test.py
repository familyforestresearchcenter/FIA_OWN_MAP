from pathlib import Path

import pandas as pd

from engine import deduplicate_table


DATA_PATH = Path(__file__).resolve().parent / "data" / "deduplication_test_data.csv"


def main():
    sample = pd.read_csv(DATA_PATH, keep_default_na=False)
    result = deduplicate_table(sample)

    checks = []
    group_to_uid = {}
    address_group_to_uid = {}

    for expected_group, group in result.groupby("expected_group"):
        group_uid_count = group["Unq_ID"].nunique()
        checks.append((f"group-{expected_group}-single-uid", group_uid_count == 1))
        group_to_uid[expected_group] = group["Unq_ID"].iloc[0]

    checks.append((
        "distinct-groups-remain-distinct",
        len(set(group_to_uid.values())) == len(group_to_uid),
    ))

    for expected_address_group, group in result.groupby("expected_address_group"):
        group_uid_count = group["Address_Unq_ID"].nunique()
        checks.append((f"address-group-{expected_address_group}-single-uid", group_uid_count == 1))
        address_group_to_uid[expected_address_group] = group["Address_Unq_ID"].iloc[0]

    checks.append((
        "distinct-address-groups-remain-distinct",
        len(set(address_group_to_uid.values())) == len(address_group_to_uid),
    ))

    initial_class_checks = {
        1: 1,
        5: 0,
        20: 3,
    }

    for test_case_id, expected_initial_class in initial_class_checks.items():
        actual_initial_class = int(result.loc[result["test_case_id"] == test_case_id, "initial_class"].iloc[0])
        checks.append((
            f"initial-class-{test_case_id}",
            actual_initial_class == expected_initial_class,
        ))

    failures = [name for name, passed in checks if not passed]

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)

    for name, _ in checks:
        print(f"PASS {name}")


if __name__ == "__main__":
    main()
