"""
data_seg.py

Goes through a folder of STARSS23 annotation CSVs and builds ONE combined
CSV containing only the rows that are:
  1. for a sound event class you care about, AND
  2. monophonic (i.e. that frame has exactly one active event total in its
     source file, so no overlapping events of any class).

Consecutive monophonic target-class frames are then collapsed into
segments, and only the START and END frame of each segment is kept
(a segment of length 1 keeps just that one frame). This avoids storing
every near-duplicate frame in a long stretch of the same active event.

Each kept row keeps all original columns, plus the source filename so you
can trace it back to the original recording.

Expected input CSV format (no header row):
    frame_number, active_class_index, source_number_index, azimuth, elevation, distance

Usage:
    python data_seg.py /path/to/annotation_folder --classes 0 3 8 --out combined_monophonic.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "frame_number",
    "active_class_index",
    "source_number_index",
    "azimuth",
    "elevation",
    "distance",
]


def keep_segment_boundaries(df):
    """
    Collapse consecutive frames into segments and keep only the first and
    last row of each segment.

    A "segment" is a run of consecutive frame_number values (step of 1)
    that share the same source_file and active_class_index. Whenever the
    frame number jumps by more than 1, or the file/class changes, a new
    segment starts.
    """
    df = df.sort_values(["source_file", "active_class_index", "frame_number"]).reset_index(drop=True)
    boundary_rows = []

    for _, group in df.groupby(["source_file", "active_class_index"]):
        frames = group["frame_number"].to_numpy()
        breaks = np.where(np.diff(frames) != 1)[0]        # indices where a segment ends
        starts = np.insert(breaks + 1, 0, 0)               # first row of each segment
        ends = np.append(breaks, len(frames) - 1)           # last row of each segment

        for start_i, end_i in zip(starts, ends):
            boundary_rows.append(group.iloc[start_i])
            if end_i != start_i:                            # avoid duplicating a length-1 segment
                boundary_rows.append(group.iloc[end_i])

    return pd.DataFrame(boundary_rows).reset_index(drop=True)


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

        # A frame is monophonic if it has exactly one event total in this file.
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
    combined = keep_segment_boundaries(combined)

    combined.to_csv(args.out, index=False)
    print(f"Saved {len(combined)} segment-boundary rows to {args.out}")


if __name__ == "__main__":
    main()