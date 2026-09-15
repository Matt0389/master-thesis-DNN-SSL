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

# Extract a single clip from start-stop boundaries
sample_idx = 0
audio, sr = frame_extract(train_df.iloc[sample_idx], train_df.iloc[sample_idx+1], audio_dir=data_path)

#%%
# Calculate STFT and plot
window = signal.get_window("hann", 128)
frame_stft = signal.ShortTimeFFT(window, int(len(window)/2), sr)

S_xy = frame_stft.spectrogram(audio[:, 0])
S_db = 10 * np.log10(S_xy)

t = np.arange(0, 0.1+frame_stft.delta_t, frame_stft.delta_t)

#%%
plt.figure()
plt.pcolormesh(t, frame_stft.f, S_db, shading="nearest")
plt.title(f"STFT of training frame {sample_idx}")
plt.show()

#%%

ps = melspectrogram(y=audio[:, 0], sr=sr, n_fft=len(window), hop_length=int(len(window)/2), window=window, n_mels=32)
ps_db = power_to_db(ps, ref=np.max)
specshow(ps_db, x_axis='s', y_axis='log')


# %%
audio_length = train_df.iloc[sample_idx+1]["frame_number"] - train_df.iloc[sample_idx]["frame_number"] + 1

t_audio = np.arange(0, (audio_length)*0.1, 1/sr)
plt.figure()
plt.plot(t_audio, audio)
plt.show()


# %%
