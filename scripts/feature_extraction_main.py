#%% IMPORTS ----------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from src.audio_tools import frame_extract, audio_to_stft, stft_to_d_r_power

import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from nara_wpe.wpe import wpe_v8
from nara_wpe.utils import stft, istft
from IPython.display import Audio

#%% LOAD AUDIO SEGMENT FROM FRAME START AND END ----------------------
data_path = "../data/STARSS23/MIC/mic_dev/dev-train-tau/"
annotation_path = "../data/STARSS23/MIC/boundary_female_speech_frames_train_tau.csv"

train_df = pd.read_csv(annotation_path)

segment = 0
audio, sr = frame_extract(segment, train_df, data_path=data_path)

audio_length = len(audio) / sr
print(audio_length)


num_cols = 9 # Number of columns within each frame gives the hop size
HOP_SIZE = int(sr*0.1/num_cols) 
print(HOP_SIZE)

#%% PLOT AUDIO CLIP FROM FRAME START AND END ------------------------
t_audio = np.arange(len(audio)) / sr
plt.figure()
plt.plot(t_audio, audio)
plt.show()


#%% CALULATE STFT AND PLOT AS COLORMESH -----------------------------

channel = 0

S_xy, SFT = audio_to_stft(audio.T, sr, win_type='hann', win_len=1024, hop=HOP_SIZE, mfft=1024)
S_db = 10 * np.log10(S_xy + 1e-10)

plt.figure()
plt.pcolormesh(S_db[0].real)
plt.title(f"STFT of training segment {segment}, channel {channel}")
plt.show()
 

# %% CALCULATE DIRECT (D) AND REVERBERRANT (R) PARTS ---------------

taps = 35
modeled_duration = taps * HOP_SIZE/sr
print(modeled_duration)

D_stft, R_stft = stft_to_d_r_power(S_xy, taps=taps, delay=3, iterations=5, mode='independent')

P_D = np.abs(D_stft)**2
P_R = np.abs(R_stft)**2

## LOG POWER PLOTTING
plt.figure()
plt.pcolormesh(10*np.log10(P_D[0] + 1e-10), shading='nearest')
plt.title(f'Log-power stft of the direct part')
plt.show()

plt.figure()
plt.pcolormesh(10*np.log10(P_R[0] + 1e-10), shading='nearest')
plt.title(f'Log-power stft of the reverberrant part')
plt.show()

#%% PLAY THE AUDIO TO VALIDATE QUALITATIVELY

print('Recorded audio')
Audio(audio.T, rate=sr)


#%%
print('Direct audio')
Audio(SFT.istft(D_stft), rate=sr)


#%% 
print('Reverberrant audio')
Audio(SFT.istft(R_stft), rate=sr)

#%% CALCULATE THE SHORT TIME POWER OF THE AUTOCORRELATION COEFFICIENTS


