#%%

from src.audio_tools import frame_extract, compute_D_and_R

import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from IPython.display import Audio
from librosa import power_to_db
from librosa.feature import melspectrogram
from librosa.display import specshow

data_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"

#annotation_path = "data/STARSS23/MIC/speech_frames_train_tau.csv"
annotation_path = "data/STARSS23/MIC/boundary_male_speech_frames_train_tau.csv"

# Import the training frame annotation file
train_df = pd.read_csv(annotation_path)

# PLOT AUDIO CLIP FROM FRAME START AND END
sample_idx = 0
audio, sr = frame_extract(train_df.iloc[sample_idx], train_df.iloc[sample_idx+1], audio_dir=data_path)

#%%

# CALULATE STFT AND PLOT AS COLORMESH 
window = signal.get_window("hann", 512)
frame_stft = signal.ShortTimeFFT(window, int(len(window)/2), sr, mfft=512)

S_xy = frame_stft.spectrogram(audio[:, 0])
S_db = 10 * np.log10(S_xy)

t = np.arange(0, 0.1+frame_stft.delta_t, frame_stft.delta_t)

print(S_db.shape)

#%%
plt.figure()
plt.pcolormesh(S_db, shading="nearest")
plt.title(f"STFT of training segment {sample_idx}")
plt.show()


# %%

# PLOT AUDIO CLIP FROM FRAME START AND END
t_audio = np.arange(len(audio)) / sr
plt.figure()
plt.plot(t_audio, audio)
plt.show()


# %%
