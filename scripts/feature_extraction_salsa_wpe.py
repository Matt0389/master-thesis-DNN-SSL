#%% IMPORTS ----------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from src.audio_tools import frame_extract, audio_to_stft, stft_to_d_r_power

import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa

from nara_wpe.wpe import wpe_v8
from nara_wpe.utils import stft, istft
from IPython.display import Audio

#%% LOAD AUDIO SEGMENT FROM FRAME START AND END ----------------------
data_path = "../data/STARSS23/MIC/mic_dev/dev-train-tau/"
annotation_path = "../data/STARSS23/MIC/boundary_male_speech_frames_train_tau.csv"

train_df = pd.read_csv(annotation_path)

segment = 0
audio, sr = frame_extract(segment, train_df, data_path=data_path)

audio_length = len(audio) / sr
print(audio_length)

HOP_SIZE = int(sr*0.1/4) # The hop size should be constant across all features and perfectly match the size of a frame
print(HOP_SIZE)

#%% PLOT AUDIO CLIP FROM FRAME START AND END ------------------------
t_audio = np.arange(len(audio)) / sr
plt.figure()
plt.plot(t_audio, audio)
plt.show()


#%% CALULATE STFT AND PLOT AS COLORMESH -----------------------------

channel = 0

S_xy, SFT = audio_to_stft(audio.T, sr, win_type='hann', win_len=512, hop=HOP_SIZE, mfft=512)
S_db = 10 * np.log10(S_xy + 1e-10)

plt.figure()
plt.pcolormesh(S_db[0].real)
plt.title(f"STFT of training segment {segment}, channel {channel}")
plt.show()
 

# %% CALCULATE DIRECT (D) AND REVERBERRANT (R) PARTS ---------------

taps = 5
modeled_duration = taps * HOP_SIZE/sr
print(modeled_duration)

P_D, P_R = stft_to_d_r_power(S_xy, taps=taps, delay=3, iterations=5, mode='independent')

## LOG POWER PLOTTING
plt.figure()
plt.pcolormesh(10*np.log10(P_D[0] + 1e-10), shading='nearest')
plt.title(f'Log-power stft of the direct signal')
plt.show()

plt.figure()
plt.pcolormesh(10*np.log10(P_R[0] + 1e-10), shading='nearest')
plt.title(f'Log-power stft of the reverberrant signal')
plt.show()

#%% CALCULATE THE SHORT TIME POWER OF THE AUTOCORRELATION COEFFICIENTS


