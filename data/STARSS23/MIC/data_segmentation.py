"""
filter_monophonic_frames.py

by Claude 9/9-26


Goes through a folder of STARSS23 annotation CSVs and builds ONE combined
CSV containing only the rows that are:
  1. for a sound event class you care about, AND
  2. monophonic (i.e. that frame has exactly one active event total in its
     source file, so no overlapping events of any class).

Each kept row keeps all original columns, plus the source filename so you
can trace it back to the original recording.

Expected input CSV format (no header row):
    frame_number, active_class_index, source_number_index, azimuth, elevation, distance

Usage:
    python filter_monophonic_frames.py /path/to/annotation_folder --classes 0 3 8 --out combined_monophonic.csv
"""

import argparse
from pathlib import Path

import pandas as pd

COLUMN_NAMES = [
    "frame_number",
    "active_class_index",
    "source_number_index",
    "azimuth",
    "elevation",
    "distance",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing annotation CSV files.")
    parser.add_argument("--classes", type=int, nargs="+", required=True, help="Target class index/indices.")
    parser.add_argument("--out", type=str, default="combined_monophonic.csv", help="Output CSV path.")
    args = parser.parse_args()

    folder = Path(args.folder)
    target_classes = set(args.classes)
    kept_rows = []

    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)

        # A frame is monophonic if it has exactly one event total in this file. !!!
        event_counts = df.groupby("frame_number").size()
        monophonic_frames = set(event_counts[event_counts == 1].index)

        # Keep rows that are both monophonic and one of the target classes.
        subset = df[
            df["frame_number"].isin(monophonic_frames)
            & df["active_class_index"].isin(target_classes)
        ].copy()

        subset.insert(0, "source_file", csv_path.name)
        kept_rows.append(subset)

    combined = pd.concat(kept_rows, ignore_index=True) if kept_rows else pd.DataFrame(
        columns=["source_file"] + COLUMN_NAMES
    )
    combined.to_csv(args.out, index=False)
    print(f"Saved {len(combined)} monophonic, target-class rows to {args.out}")


if __name__ == "__main__":
    main()