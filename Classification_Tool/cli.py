import argparse
import pandas as pd
from engine import classify_owner, Trace


def main():
    parser = argparse.ArgumentParser("Ownership classifier")

    parser.add_argument("own1", nargs="?", help="Single owner name")
    parser.add_argument("--own2")
    parser.add_argument("--state")

    parser.add_argument("--csv", help="Input CSV path")
    parser.add_argument("--out", help="Output CSV path")

    # NEW FLAG
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable trace logging"
    )

    args = parser.parse_args()

    # -------------------------
    # CSV mode
    # -------------------------

    if args.csv:
        df = pd.read_csv(args.csv)
        out = classify_table(df)

        if args.out:
            out.to_csv(args.out, index=False)
        else:
            print(out.head())

        return

    # -------------------------
    # Single-name mode
    # -------------------------

    if args.own1:

        trace = Trace(args.debug)

        own_type, trace = classify_owner(
            args.own1,
            own2=args.own2,
            state_name=args.state,
            trace=trace
        )

        print(f"Result:{own_type}")

        if args.debug:
            print("\nTrace log:")
            for event in trace.events:
                print(event)

        return

    parser.error("Provide either a single owner or --csv")


if __name__ == "__main__":
    main()