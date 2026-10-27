
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class SSL_Dataset(Dataset):

    def __init__(self, feature_dir, label_dir, chunk_seconds=4.0, feature_hop_sec=0.0125,
                 hop_frames=None, frames_per_annotation=8, target_classes=(0, 1)):
        # The function needs directories for the features and the labels. 
        # The target classes is because I want my system to perform primarily for speech, and passing this argument allows the model to consider non-speech classes as distractors in polyphonic situations.

        super().__init__()

        self.target_classes = set(target_classes)

        # Grab features and labels
        self.feature_paths = sorted(Path(feature_dir).glob("*.npy"))
        self.label_paths = [Path(label_dir) / f"{p.stem}.csv" for p in self.feature_paths]

        # define the chunks as feature columns instead (chunk length in seconds divided by feature resolution, which is 12.5ms)
        self.chunk_frames = int(round(chunk_seconds / feature_hop_sec))

        self.frames_per_annotation = frames_per_annotation

        # Distance (in feature frames) between consecutive chunk starts. Defaults to
        # non-overlapping chunks (hop == chunk length) when the caller doesn't set one.
        # I might want to play with this at some point
        self.hop_frames = hop_frames or self.chunk_frames


        # For making sure each annotation frame matches cleanly to a feature length of 8 columns per input
        assert self.chunk_frames % frames_per_annotation == 0, (
            f"chunk_frames ({self.chunk_frames}) must be a multiple of "
            f"frames_per_annotation ({frames_per_annotation})."
        )
        assert self.hop_frames % frames_per_annotation == 0, (
            f"hop_frames ({self.hop_frames}) must be a multiple of "
            f"frames_per_annotation ({frames_per_annotation}), so every chunk start "
            f"lines up with an annotation-row boundary."
        )


        # Number of (block, prediction) units per chunk -- e.g. 320 feature frames / 8 = 40 annotation blocks
        self.blocks_per_chunk = self.chunk_frames // frames_per_annotation

        # (recording_idx, start_frame) for every valid training chunk
        self.index = []      
         # one dict per recording: {annotation_row: (azimuth, elevation, distance)}   
        self.label_lookup = []  


        # Walk every recording once, up front, to (a) record which chunks are valid and (b) build a fast per-recording label lookup -- both done here so __getitem__ later does no file-format parsing, just array indexing.
        for rec_idx, feat_path in enumerate(self.feature_paths):

            # mmap_mode="r" reads only the .npy header (shape, dtype) here, not the
            # feature data itself -- this line does not load the recording into RAM.
            n_frames = np.load(feat_path, mmap_mode="r").shape[1]
 
            if n_frames < self.chunk_frames:
                # This recording is shorter than one chunk, so it can never produce a
                # full chunk -- contributes nothing to self.index. Still append an
                # empty lookup so self.label_lookup stays aligned 1:1 with
                # self.feature_paths by rec_idx (every recording gets exactly one entry).
                self.label_lookup.append({})
                continue
 
            # Latest feature frame a chunk could start at and still fit inside this
            # recording -- e.g. 120 frames total, 40-frame chunks -> last_start = 80.
            last_start = n_frames - self.chunk_frames
 
            # Record one (recording, start) pair per valid chunk, stepping by
            # hop_frames each time (non-overlapping chunks by default).
            for start in range(0, last_start + 1, self.hop_frames):
                self.index.append((rec_idx, start))


            # Read the label csv file
            df = pd.read_csv(
            self.label_paths[rec_idx],
            header=None,
            names=["frame_number", "active_class_index", "source_number_index",
                    "azimuth", "elevation", "distance"],
        )

            # This loop handles polyphonic frames, removing the labels associated with non-target classes. This way, the dataset fascilitates speech-only detection and localization.
            lookup = {}

            for frame_number, group in df.groupby("frame_number"):
                target_rows = group[group["active_class_index"].isin(self.target_classes)]
                if len(target_rows) == 0:
                    continue  # only distractor class(es) active here -- no target, leave unset
 
                # If more than one target class is simultaneously active (e.g. two
                # overlapping speakers), this still only keeps the first -- genuine
                # multi-speaker overlap isn't handled yet, just speech-vs-distractor.
                row = target_rows.iloc[0]
                lookup[int(frame_number)] = (
                    float(row["azimuth"]),
                    float(row["elevation"]),
                    float(row["distance"]),
                )
            self.label_lookup.append(lookup)


    def __len__(self):

        # Self-explanatory

        return len(self.index)




    def __getitem__(self, index):

        # index: an integer from 0 up to len(self) - 1, supplied by the DataLoader (in random order, if shuffle=True)


        rec_idx, start = self.index[index]
        end = start + self.chunk_frames        # exclusive end of this chunk, in FEATURE frames
        lookup = self.label_lookup[rec_idx]    # this recording's {annotation_row: (azi, ele, dist)}
 

        # Features stay DENSE, at the full feature rate -- the model's internal layers still get to see fine 12.5ms resolution. Only the labels/outputs below are coarsened to annotation rate, matching one output every 100ms.

        feats = np.load(self.feature_paths[rec_idx], mmap_mode="r")[:, start:end, :]  # (Channels, chunk_frames, F)
 

        # One label slot per annotation block -- e.g. 40 for a 4-second chunk

        first_anno = start // self.frames_per_annotation
        y = np.zeros((self.blocks_per_chunk, 3), dtype=np.float32)
        active = np.zeros(self.blocks_per_chunk, dtype=bool)
 
        for i in range(self.blocks_per_chunk):
            values = lookup.get(first_anno + i)  # None if this annotation row has no entry (silence)
            if values is not None:
                y[i] = values        # this block's target -- no repeating needed anymore
                active[i] = True     # and this block counts as active
 
        # Convert everything to torch tensors for the model/loss to consume.
        # np.ascontiguousarray guards against the slice above leaving an awkward
        # memory layout that torch.from_numpy would otherwise reject.
        x = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))  # (chunk_frames, C, F)
        y = torch.from_numpy(y)                              # (blocks_per_chunk, 3)
        mask = torch.from_numpy(active.astype(np.float32))   # (blocks_per_chunk,)
 
        return x, y, mask


