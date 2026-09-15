import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
from pathlib import Path

# STARSS23 annotations are given at a 10 Hz frame rate -> 100 ms per frame.
FRAME_HOP_SEC = 0.1
 
 
def frame_extract(start_row, end_row, frame_hop_sec=FRAME_HOP_SEC, audio_dir=None):
    """
    This function is for extracting an audio clip spanning start_row to end_row (inclusive).
    """
    assert start_row["source_file"] == end_row["source_file"], "start/end frames must be from the same file"
 
    fname = Path(audio_dir or ".") / start_row["source_file"]
    fname = fname.with_suffix(".wav")
    sr = sf.info(fname).samplerate
 
    start_sample = int(start_row["frame_number"] * frame_hop_sec * sr)
    end_sample = int((end_row["frame_number"] + 1) * frame_hop_sec * sr)
 
    return sf.read(fname, start=start_sample, frames=end_sample - start_sample, always_2d=True)
 





def compute_D_and_R(mic_signals, fs, taps=10, delay=3, iterations=3):
    """
    Function for estimating the direct and reverberrant parts of the audio file, to use as one of the input features for distance estimaation. 
    mic_signals: (num_mics, num_samples) — raw time-domain capsule signals
    Returns: P_D, P_R — power spectra, shape (num_mics, F, T)
    """
    # 1. STFT each raw capsule channel
    _, _, Y = signal.stft(mic_signals, fs=fs, nperseg=512, noverlap=384, axis=-1)
    # Y shape: (num_mics, F, T)

    # 2. nara_wpe expects (F, D, T) — move channel axis
    Y_wpe = np.transpose(Y, (1, 0, 2))

    # 3. Run WPE — this alone gives you D directly, no extra step
    D_stft = wpe(Y_wpe, taps=taps, delay=delay, iterations=iterations)  # (F, D, T)

    # 4. Interception point: R is just the leftover, still in STFT domain
    R_stft = Y_wpe - D_stft   # valid because STFT is linear

    # 5. Power spectra — same ε-floor as Berghi to avoid log(0) downstream
    P_D = np.abs(D_stft) ** 2
    P_R = np.abs(R_stft) ** 2
    return P_D, P_R