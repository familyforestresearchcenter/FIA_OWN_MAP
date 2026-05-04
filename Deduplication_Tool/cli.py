import argparse

from engine import deduplicate_csv


def main():
    parser = argparse.ArgumentParser("Ownership deduplication")
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV path; expected name field is FIRST_LABEL_LINE and address fields are ADDRESS1, CITY, STATEAB, ZIP_CD",
    )
    parser.add_argument("--out", help="Output CSV path")

    args = parser.parse_args()

    output = deduplicate_csv(args.csv, args.out)

    if not args.out:
        print(output.head())


if __name__ == "__main__":
    main()
