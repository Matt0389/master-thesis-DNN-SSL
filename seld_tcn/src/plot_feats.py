

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


file = "seld_tcn/features/fold3_room12_mix002.npy"
feats = np.load(file)


# TF conversion for plotting
FS = 24000
N_FFT = 512
HOP_LEN = 300
channel = 0



n_time_frames, n_freq_bins = feats.shape[1], feats.shape[2]

freqs = np.arange(n_freq_bins) * FS / N_FFT
times = np.arange(n_time_frames) * HOP_LEN / FS

extent = [times[0], times[-1], freqs[0], freqs[-1]]

LinSpec = feats[channel, :, :]
CDPD = feats[4, :, :]
SALSA = feats[channel + 5, :, :]

names = [["LinSpec", channel, "dB"], ["CDPD", 4, "Nominal"], ["SALSA", channel + 5, "RDOA (m)"]]

for i in range(3):

    plt.figure(figsize=(8, 5))
    plt.imshow(feats[names[i][1]].T, extent=extent, cmap="nipy_spectral", interpolation="Nearest", origin="lower", aspect="auto")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.xlim(3, 7)
    cbar = plt.colorbar()
    cbar.set_label(names[i][2])

    title = f"{os.path.basename(file)}" # - mean of azi, ele, dist: ({means["azimuth"]}, {means["elevation"]}, {means["distance"]})"

    plt.title(title)

    plt.savefig("seld_tcn/" + f"{os.path.basename(file)}-{i}.png")
