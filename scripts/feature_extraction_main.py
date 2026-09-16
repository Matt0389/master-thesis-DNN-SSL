#%% IMPORTS

from src.audio_tools import frame_extract, compute_d_r_power

import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from nara_wpe.wpe import wpe_v8
from nara_wpe.utils import stft, istft
from IPython.display import Audio
from librosa import power_to_db
from librosa.feature import melspectrogram
from librosa.display import specshow


#%% LOAD AUDIO SEGMENT FROM FRAME START AND END
data_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"
annotation_path = "data/STARSS23/MIC/boundary_male_speech_frames_train_tau.csv"

# Import the training frame annotation file
train_df = pd.read_csv(annotation_path)

sample_idx = 0
audio, sr = frame_extract(train_df.iloc[sample_idx], train_df.iloc[sample_idx+1], audio_dir=data_path)

#%% PLOT AUDIO CLIP FROM FRAME START AND END
t_audio = np.arange(len(audio)) / sr
plt.figure()
plt.plot(t_audio, audio)
plt.show()


#%% CALULATE STFT AND PLOT AS COLORMESH

channel = 0
window = signal.get_window("hann", 512)
frame_stft = signal.ShortTimeFFT(window, int(len(window)/2), sr, mfft=512)

S_xy = frame_stft.spectrogram(audio[:, channel])
S_db = 10 * np.log10(S_xy)

t = np.arange(0, 0.1+frame_stft.delta_t, frame_stft.delta_t)

print(S_db.shape)

plt.figure()
plt.pcolormesh(S_db, shading="nearest")
plt.title(f"STFT of training segment {sample_idx}, channel {channel}")
plt.show()
 

# %% CALCULATE DIRECT (D) AND REVERBERRANT (R) PARTS

P_D, P_R, SFT = compute_d_r_power(audio.T, sr)

## LOG POWER PLOTTING
plt.figure()
plt.pcolormesh(10*np.log10(P_D[0] + 1e-10), shading='nearest')
plt.title(f'Log-power stft of the direct signal')
plt.show()

plt.figure()
plt.pcolormesh(10*np.log10(P_R[0] + 1e-10), shading='nearest')
plt.title(f'Log-power stft of the reverberrant signal')
plt.show()

# %%
