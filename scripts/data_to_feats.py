import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio_tools import frame_extract, audio_to_stft
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
    # data_flag = "Real "
    # data_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"
    # annotation_path = "data/STARSS23/MIC/tau_train_female_speech_segments.csv"

    # train_df = pd.read_csv(annotation_path)

    # segment = 41 # Which clip to import
    # # audio_data = frame_extract(segment, train_df, data_path=data_path, sr=FS, mono=False, dtype=np.float32).T

    # row = train_df.iloc[segment]

    # fname = Path(data_path or ".") / row["source_file"]
    # fname = fname.with_suffix(".wav")

    # audio_data, _ = librosa.load(fname, sr=16000, mono=False, offset=row["frame_number"]*0.1, duration=row["segment_length"]*0.1)

    # # means = pd.read_csv("data/STARSS23/MIC/metadata_dev/dev-train-tau/" + row["source_file"], header=None, names=COLUMN_NAMES, usecols=["azimuth", "elevation", "distance"]).iloc[row["frame_number"] : row["frame_number"] + row["segment_length"]].mean()

    # # salsadlite_feat = get_salsa_dlite(audio_data)
    # salsadlite_feat = _get_salsa(audio_data, add_cdpd=True)
    # ---------------------------------------
    
    
    
    ## ========== Synthetic data ===========
    #------------------------------------
    data_flag = "Synthetic "
    synth_mov = "fold1_sc203_mix031" # 7, 8
    synth_static = "fold1_sc203_mix034" # 1, 8

    data_path = "data/synth/output/mic/" + synth_static + ".wav"
    annotation_path = "data/synth/output/labels/" + synth_static + ".csv"

    audio_data_synth, _ = librosa.load(data_path, sr=16000, mono=False, offset=1, duration=8)

    # salsadlite_feat = get_salsa_dlite(audio_data_synth)
    salsadlite_feat = _get_salsa(audio_data_synth, add_cdpd=True)
    #-----------------------------------

    
    
    print(f"SALSA-DLite: {salsadlite_feat.shape}")

    

    #========= PLOTTING

    # TF conversion for plotting
    FS = 16000
    N_FFT = 512
    HOP_LEN = 200
    channel = 0


    # stft, SFT = audio_to_stft(audio_data_synth, FS, hop=HOP_LEN)

    # plt.figure()
    # plt.imshow(np.abs(stft[channel]), origin="lower")
    # plt.savefig(f"data/{data_flag}_logspec.png")


    n_time_frames, n_freq_bins = salsadlite_feat.shape[1], salsadlite_feat.shape[2]

    freqs = np.arange(n_freq_bins) * FS / N_FFT
    times = np.arange(n_time_frames) * HOP_LEN / FS

    extent = [times[0], times[-1], freqs[0], freqs[-1]]

    LinSpec = salsadlite_feat[channel, :, :].T
    CDPD = salsadlite_feat[4, :, :].T,
    SALSA = salsadlite_feat[channel + 5, :, :].T
    
    names = [["LinSpec", channel, "dB"], ["CDPD", 4, "Nominal"], ["SALSA", channel + 5, "RDOA (m)"]]

    for i in range(3):

        plt.figure(figsize=(8, 5))
        plt.imshow(salsadlite_feat[names[i][1]].T, extent=extent, cmap="nipy_spectral", interpolation="Nearest", origin="lower", aspect="auto")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        cbar = plt.colorbar()
        cbar.set_label(names[i][2])

        title = data_flag + f"recorded static female speech - {names[i][0]}" # - mean of azi, ele, dist: ({means["azimuth"]}, {means["elevation"]}, {means["distance"]})"
    
        plt.title(title)

        plt.savefig("data/" + data_flag + f"recorded female speech static source - {names[i][0]}.png")
    
    print(np.__version__, librosa.__version__)