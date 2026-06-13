"""Create a CSV manifest for the assignment data."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meg_decode.manifest import discover_records, write_manifest_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Final Project data/Final Project data")
    parser.add_argument("--output", default="outputs/manifest.csv")
    args = parser.parse_args()

    records = discover_records(args.data_root)
    write_manifest_csv(records, args.output)
    counts = Counter((record.protocol, record.split, record.class_name) for record in records)
    print(f"Wrote {len(records)} records to {Path(args.output)}")
    for key, count in sorted(counts.items()):
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
