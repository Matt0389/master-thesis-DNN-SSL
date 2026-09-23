"""
data_seg.py

Goes through a folder of STARSS23 annotation CSVs and builds ONE combined
CSV of clean, monophonic single-class segments, ready for training.

For each file, extract_monophonic_segments():
  1. Filters to frames that are monophonic (exactly one active event total
     in that file, across all classes - so no overlapping events).
  2. Filters to your target sound event class(es).
  3. Collapses the surviving frames into segments: consecutive frames
     (within gap_tolerance_sec of each other) become one segment, keeping
     only the segment's START frame plus a "segment_length" column giving
     how many frames it spans. A larger gap, or a class change, starts a
     new segment. This avoids storing every near-duplicate frame in a long
     stretch of the same active event, while still tolerating brief gaps
     (e.g. short silences) without splitting what is really one clip.

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

FRAME_HOP_SEC = 0.1              # 10 Hz STARSS23 annotation rate
SEGMENT_GAP_TOLERANCE_SEC = 0.1  # bridge gaps up to this many seconds into one segment


def extract_monophonic_segments(df, target_classes, source_file,
                                 gap_tolerance_sec=SEGMENT_GAP_TOLERANCE_SEC,
                                 frame_hop_sec=FRAME_HOP_SEC):
    """
    Given one file's raw annotation dataframe (all classes, all frames),
    return one row per clean segment: monophonic, one of target_classes,
    with consecutive frames (within gap_tolerance_sec) collapsed into a
    single row carrying frame_number (segment start) and segment_length
    (how many frames it spans - the segment covers frame_number ...
    frame_number + segment_length - 1 inclusive).
    """
    # A frame is monophonic if it has exactly one event total in this file.
    event_counts = df.groupby("frame_number").size()
    monophonic_frames = set(event_counts[event_counts == 1].index)

    subset = df[
        df["frame_number"].isin(monophonic_frames)
        & df["active_class_index"].isin(target_classes)
    ].copy()
    subset.insert(0, "source_file", source_file)

    if subset.empty:
        return subset.assign(segment_length=pd.Series(dtype=int))

    gap_tolerance_frames = gap_tolerance_sec / frame_hop_sec
    subset = subset.sort_values(["active_class_index", "frame_number"]).reset_index(drop=True)

    segment_rows = []
    for _, group in subset.groupby("active_class_index"):
        frames = group["frame_number"].to_numpy()
        breaks = np.where(np.diff(frames) > gap_tolerance_frames)[0]  # indices where a segment ends
        starts = np.insert(breaks + 1, 0, 0)
        ends = np.append(breaks, len(frames) - 1)

        for start_i, end_i in zip(starts, ends):
            row = group.iloc[start_i].copy()
            row["segment_length"] = int(frames[end_i] - frames[start_i] + 1)
            segment_rows.append(row)

    return pd.DataFrame(segment_rows).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing annotation CSV files.")
    parser.add_argument("--classes", type=int, nargs="+", required=True, help="Target class index/indices.")
    parser.add_argument("--gap-tolerance", type=float, default=SEGMENT_GAP_TOLERANCE_SEC,
                         help=f"Max gap in seconds to bridge within one segment (default {SEGMENT_GAP_TOLERANCE_SEC}).")
    parser.add_argument("--out", type=str, default="combined_monophonic.csv", help="Output CSV path.")
    args = parser.parse_args()

    folder = Path(args.folder)
    target_classes = set(args.classes)
    all_segments = []

    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
        segments = extract_monophonic_segments(
            df, target_classes, csv_path.name, gap_tolerance_sec=args.gap_tolerance
        )
        all_segments.append(segments)

    combined = pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame(
        columns=["source_file"] + COLUMN_NAMES + ["segment_length"]
    )

    combined.to_csv(args.out, index=False)
    print(f"Saved {len(combined)} segments to {args.out}")


if __name__ == "__main__":
    main()