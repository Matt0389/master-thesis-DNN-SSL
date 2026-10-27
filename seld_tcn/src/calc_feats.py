

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.extract_salsa_feats import get_salsa_dlite, _get_salsa

import numpy as np
import pandas as pd
import glob
import librosa



dev_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"
sound_files = glob.glob(dev_path + "*.wav", recursive=True)

feat_path = "seld_tcn/features"

FS = 24000


for file in sound_files:

    fname = os.path.splitext(os.path.basename(file))[0]

    audio_data, _ = librosa.load(file, sr=24000, mono=False)

    audio_len = int(len(audio_data[0, :]) / FS)

    salsad_feat = _get_salsa(audio_data=audio_data, add_cdpd=True)

    np.save(os.path.join(feat_path, fname + ".npy"), salsad_feat)

    print(f"Completed feature extraction for file {fname} - Feature shape: {salsad_feat.shape} - Audio length: {audio_len}")

