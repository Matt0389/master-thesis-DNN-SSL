import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd
import soundfile as sf

from src.extract_salsa_feats import get_salsa_dlite

# ---------------- config ----------------
data_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"
annotation_path = "data/STARSS23/MIC/tau_train_female_speech_segments.csv"
raw_anno_source_path = "data/STARSS23/MIC/metadata_dev/dev-train-tau/"

OUT_PATH = Path("data/STARSS23/features/salsa_dlite.h5")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FS = 16000
Ts_PER_ANNOTATION = 8        # feature frames per 100 ms annotation row
FRAME_HOP_SEC = 0.1
CHUNK_SECONDS = 2            # HDF5 chunk length -- match your typical read size
MAX_CLIPS = 50             # set an int to cap rows for a quick test run

# ---------------- pass over clips, accumulate in memory ----------------
segment_df = pd.read_csv(annotation_path)

feat_list, label_list, clip_ids, n_frames = [], [], [], []

for idx, segment in segment_df.iterrows():
    if MAX_CLIPS is not None and idx >= MAX_CLIPS:
        break

    fname = data_path + Path(segment["source_file"]).stem + ".wav"
    anno_fname = raw_anno_source_path + segment["source_file"]
    file_anno_df = pd.read_csv(
        anno_fname, header=None,
        names=["frame_number", "active_class_index", "source_number_index",
               "azimuth", "elevation", "distance"],
    )

    start_sample = int(segment["frame_number"] * FRAME_HOP_SEC * FS)
    end_sample = int((segment["frame_number"] + segment["segment_length"]) * FRAME_HOP_SEC * FS)

    audio, _ = sf.read(fname, start=start_sample, frames=end_sample - start_sample,
                        always_2d=True, dtype="float32")
    audio = audio.T  # (channels, samples)

    features = get_salsa_dlite(audio)             # (C, T, F)
    features = np.transpose(features, (1, 0, 2))  # -> (T, C, F): frame-major, matches reader
    t_feat = features.shape[0]

    anno = file_anno_df.loc[
        segment["frame_number"]: segment["frame_number"] + segment["segment_length"] - 1,
        ["azimuth", "elevation", "distance"],
    ].to_numpy()

    labels = np.repeat(anno, Ts_PER_ANNOTATION, axis=0)

    if len(labels) < t_feat:

        labels = np.pad(labels, ((0, t_feat - len(labels)), (0, 0)))

    labels = labels[:t_feat].astype(np.float32)

    clip_id = f"{Path(segment['source_file']).stem}_{segment['frame_number']}"
    feat_list.append(features.astype(np.float32))
    label_list.append(labels)
    clip_ids.append(clip_id)
    n_frames.append(t_feat)

    print(f"[{idx}] {clip_id}: {t_feat} frames")

# ---------------- store data as a HDF5 file ----------------
all_feats = np.concatenate(feat_list, axis=0)      # (total_T, C, F)
all_labels = np.concatenate(label_list, axis=0)    # (total_T, 3)
bounds = np.concatenate([[0], np.cumsum(n_frames)]).astype(np.int64)

chunk_len = min(int(CHUNK_SECONDS / FRAME_HOP_SEC) * Ts_PER_ANNOTATION, all_feats.shape[0])

with h5py.File(OUT_PATH, "w") as f:
    f.create_dataset("features", data=all_feats,
                      chunks=(chunk_len, *all_feats.shape[1:]))
    f.create_dataset("labels", data=all_labels,
                      chunks=(chunk_len, all_labels.shape[1]))
    f.create_dataset("recording_bounds", data=bounds)
    f.create_dataset("clip_ids", data=np.array(clip_ids, dtype=h5py.string_dtype()))
    f.attrs["fs"] = FS
    f.attrs["frame_hop_sec"] = FRAME_HOP_SEC

print(f"Wrote {all_feats.shape[0]} frames from {len(clip_ids)} clips to {OUT_PATH}")