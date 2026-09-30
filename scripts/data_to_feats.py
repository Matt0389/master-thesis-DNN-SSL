import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio_tools import frame_extract
from src.extract_salsa_feats import get_salsa_dlite, _get_salsa

import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import pandas as pd

FS=16000 ## Change to 16000 in line with the synthetic dataset

#================
#   Main
#================



if __name__ == "__main__":

    # ============== extract target segment =====================

    COLUMN_NAMES = ["frame_number", "active_class_index", "source_number_index", "azimuth", "elevation", "distance"]
    

    # ## ========== Real data ===========

    # ---------------------------------------
    data_flag = "Real "
    data_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"
    annotation_path = "data/STARSS23/MIC/tau_train_female_speech_segments.csv"

    train_df = pd.read_csv(annotation_path)

    segment = 0 # Which clip to import
    # audio_data = frame_extract(segment, train_df, data_path=data_path, sr=FS, mono=False, dtype=np.float32).T

    row = train_df.iloc[segment]

    fname = Path(data_path or ".") / row["source_file"]
    fname = fname.with_suffix(".wav")

    audio_data, _ = librosa.load(fname, sr=16000, mono=False, offset=row["frame_number"], duration=row["segment_length"]*0.1)

    means = pd.read_csv("data/STARSS23/MIC/metadata_dev/dev-train-tau/" + row["source_file"], header=None, names=COLUMN_NAMES, usecols=["azimuth", "elevation", "distance"]).iloc[row["frame_number"] : row["frame_number"] + row["segment_length"]].mean()

    salsadlite_feat = get_salsa_dlite(audio_data)
    # ---------------------------------------
    
    
    
    ## ========== Synthetic data ===========
    #------------------------------------
    # data_flag = "Synthetic "
    # data_path = "data/synth/output/mic/fold1_metu_mix001.wav"
    # annotation_path = "data/synth/output/labels/fold1_metu_mix001.csv"

    # audio_data_synth, _ = librosa.load(data_path, sr=16000, mono=False, offset=0, duration=8)
    # # audio_data_synth = audio_data_synth.T

    # anno_df = pd.read_csv(annotation_path, header=None, names=COLUMN_NAMES, usecols=["azimuth", "elevation", "distance"])

    # anno_df["distance"] = anno_df["distance"]*100

    # means = anno_df.mean().round().astype(int)

    # salsadlite_feat = get_salsa_dlite(audio_data_synth)
    #-----------------------------------





    # audio_length = len(audio_data[0, :]) / FS
    # print(f"Audio length {audio_length} s , audio shape {audio_data.shape}")


    # =============== calculate SALSA-D-LITE feature maps ======
    #print(f"Lower Bin: {LOWER_BIN}, Upper Bin: {UPPER_BIN}, Cutoff Bin: {UPPER_BIN}, SALSA-bins: {N_SALSA_BINS}")
    # sample_audio_fp = ""
    
    
    print(f"SALSA-DLite: {salsadlite_feat.shape}")

    #========= PLOTTING

    # TF conversion for plotting
    FS = 16000
    N_FFT = 512
    HOP_LEN = 200

    n_freq_bins, n_time_frames = salsadlite_feat.shape[1], salsadlite_feat.shape[2]

    freqs = np.arange(n_freq_bins) * FS / N_FFT     # Hz per bin
    times = np.arange(n_time_frames) * HOP_LEN / FS  # seconds per frame

    extent = [times[0], times[-1], freqs[0], freqs[-1]]


    channel = 0
    maps = [
        (salsadlite_feat[channel, :, :].T, f"LinSpec for channel {channel}"),
        (salsadlite_feat[4, :, :].T, "CDPD feature map"),
        (salsadlite_feat[channel + 5, :, :].T, f"NIPD for channels {channel} and {channel+1}"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # One shared colorbar only makes sense if all three panels use the same value
    # range - CDPD/NIPD/log-magnitude are on genuinely different scales, so a
    # single vmin/vmax will likely wash out at least one of them. Worth checking
    # the plots below before trusting a single shared colorbar's contrast.
    vmin = min(m.min() for m, _ in maps)
    vmax = max(m.max() for m, _ in maps)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for ax, (data, title) in zip(axes, maps):
        im = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest", extent=extent)
        fig.colorbar(im, ax=ax, location="right", shrink=0.8)
        ax.set_title(title)

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_ylabel("Frequency (Hz)")

    
    means = means.round().astype(int)

    title = data_flag + f"recorded static female speech - mean of azi, ele, dist: ({means["azimuth"]}, {means["elevation"]}, {means["distance"]})"
    fig.suptitle(title)
    

    fig.savefig("data/" + data_flag + "recorded female speech - selected features - static source.png")
    
    print(np.__version__, librosa.__version__)