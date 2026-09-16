import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from nara_wpe.wpe import wpe_v8

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
 





def compute_d_r_power(mic_signals, fs, win_len=512, hop=128, mfft=None,
                       taps=10, delay=3, iterations=3):
    """
    mic_signals: (num_mics, num_samples) — raw time-domain capsule signals
    win_len:     window length in samples -> sets frequency resolution
    hop:         samples between frames -> sets your feature map's frame rate
    mfft:        FFT length; None defaults to win_len (no zero-padding)

    Returns
    -------
    P_D, P_R : ndarray, shape (num_mics, F, T) — direct/reverberant power spectra
    SFT      : the ShortTimeFFT object (keep it — you'll want SFT.f / SFT.delta_t
               for mel-projection and for aligning frames to your 10 Hz annotation
               boundaries later)
    """
    # 1. Build the STFT transform once. sym=True matches scipy's own
    #    ShortTimeFFT examples for analysis windows.
    win = signal.windows.hann(win_len, sym=True)
    SFT = signal.ShortTimeFFT(win, hop=hop, fs=fs, mfft=mfft, fft_mode='onesided')

    # 2. STFT all mic channels in one call — channel axis is batched automatically
    Y = SFT.stft(mic_signals, axis=-1)          # shape (num_mics, F, T)

    # 3. nara_wpe expects (F, D, T) — move the channel axis
    Y_wpe = np.transpose(Y, (1, 0, 2))

    # 4. Run WPE — the output IS the direct/early estimate, no extra step
    D_stft = wpe_v8(Y_wpe, taps=taps, delay=delay, iterations=iterations)  # (F, D, T)

    # 5. Interception point: reverberant residual, still in STFT domain
    R_stft = Y_wpe - D_stft

    # 6. Back to (num_mics, F, T) and to power
    D_stft = np.transpose(D_stft, (1, 0, 2))
    R_stft = np.transpose(R_stft, (1, 0, 2))
    P_D = np.abs(D_stft) ** 2
    P_R = np.abs(R_stft) ** 2

    return P_D, P_R, SFT