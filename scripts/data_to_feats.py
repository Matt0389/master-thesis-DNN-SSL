import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio_tools import frame_extract
from src.extract_salsa_feats import get_salsa_dlite

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FS=24000

#================
#   Main
#================

if __name__ == "__main__":

    # ============== extract target segment =====================

    data_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"
    annotation_path = "data/STARSS23/MIC/boundary_male_speech_frames_train_tau.csv"

    train_df = pd.read_csv(annotation_path)

    segment = 0
    audio_data = frame_extract(segment, train_df, data_path=data_path, sr=FS, mono=False, dtype=np.float32).T

    audio_length = len(audio_data[0, :]) / FS
    print(f"Audio length {audio_length} s , audio shape {audio_data.shape}")


    # =============== calculate SALSA-D-LITE feature maps ======
    #print(f"Lower Bin: {LOWER_BIN}, Upper Bin: {UPPER_BIN}, Cutoff Bin: {UPPER_BIN}, SALSA-bins: {N_SALSA_BINS}")
    # sample_audio_fp = ""
    
    #salsalite_feat = _get_salsalite(sample_audio_fp)
    salsadlite_feat = get_salsa_dlite(audio_data)
    #salsa_feat = _get_salsa(sample_audio_fp)
    #salsad_feat = _get_salsa(sample_audio_fp)

    print(f"SALSA-DLite: {salsadlite_feat.shape}")

    plt.figure()
    plt.imshow(salsadlite_feat[4].T, aspect='auto', origin='lower', interpolation='nearest')
    plt.title("CDPD feature map")
    plt.savefig("CDPD-fmap.png")

    channel = 0
    plt.figure()
    plt.imshow(salsadlite_feat[channel].T, aspect='auto', origin='lower', interpolation='nearest')
    plt.title(f"LinSpec for channel {channel}")
    plt.savefig(f"LinSpec for channel {channel}.png")

    plt.figure()
    plt.imshow(salsadlite_feat[channel+5].T, aspect='auto', origin='lower', interpolation='nearest')
    plt.title(f"NIPD for channels {channel} and {channel+1}")
    plt.savefig(f"channel-{channel}-{channel+1}-NIPD.png")
    
    print(np.__version__, librosa.__version__)