import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from nara_wpe.wpe import wpe_v8

# STARSS23 annotations are given at a 10 Hz frame rate -> 100 ms per frame.

 
 
def frame_extract(sample_idx=0, annotation_df=None, data_path=None, sr=24000, mono=False, dtype="float32"):
    """
    This function is for extracting an audio clip spanning from start_row and until the number of frames specified by the df row is reached.
    """
    frame_hop_sec = 0.1

    # Import the training frame annotation file

    start_row = annotation_df.iloc[sample_idx]
    audio_dir=data_path
 
    fname = Path(audio_dir or ".") / start_row["source_file"]
    fname = fname.with_suffix(".wav")

    seg_length = (start_row["frame_number"] + start_row["segment_length"]) # Length of the segment
 
    start_sample = int(start_row["frame_number"] * frame_hop_sec * sr)
    end_sample = int(seg_length * frame_hop_sec * sr)

    print(f"Final sample idx: {seg_length}")
 
    audio, _ = sf.read(fname, start=start_sample, frames=end_sample - start_sample, always_2d=True, dtype=dtype)

    return audio
 

def audio_to_stft(mic_signals, fs, win_type='hann', win_len=512, hop=512, mfft=512):
    """
    Function for calculating the stft from a (multichannel) audio signal.

    This STFT is thus both used for the SALSA-Lite feature as well as the estimation of the direct and reverberrant parts of each capsule input.  


    """

    win = signal.get_window(win_type, win_len)
    SFT = signal.ShortTimeFFT(win, hop=hop, fs=fs, mfft=mfft, fft_mode='onesided')

    # STFT all mic channels in one call
    Y = SFT.stft(mic_signals, axis=-1) # shape (num_mics, F, T)

    return Y, SFT





def stft_to_d_r_power(stft, taps=10, delay=3, iterations=3, mode='independent'):
    """
    mic_signals: (num_mics, num_samples) — raw time-domain capsule signals
    mode: 'independent' — each capsule's WPE filter only sees its own past
                           (D=1 per channel; no cross-capsule information used)
          'joint'        — MIMO WPE; each channel's filter can draw on all
                           capsules' past jointly (D=num_mics)

    Returns
    -------
    P_D, P_R : ndarray, shape (num_mics, F, T) — direct/reverberant power spectra
    SFT      : the ShortTimeFFT object
    """

    Y = stft       # Is shape (num_mics, F, T) if originating from audio_to_logstft

    # Nara_wpe expects (F, D, T) — move the channel axis
    Y_wpe = np.transpose(Y, (1, 0, 2))

    # 4. Run WPE — joint (cross-capsule) or independent (per-capsule)
    if mode == 'joint':
        D_stft = wpe_v8(Y_wpe, taps=taps, delay=delay, iterations=iterations)  # (F, D, T)

    elif mode == 'independent':
        D_stft = np.empty_like(Y_wpe)
        for m in range(Y_wpe.shape[1]):
            y_m = Y_wpe[:, m:m+1, :]                                  # (F, 1, T)
            D_stft[:, m:m+1, :] = wpe_v8(y_m, taps=taps, delay=delay,
                                          iterations=iterations)

    else:
        raise ValueError(f"mode must be 'joint' or 'independent', got {mode!r}")

    # 5. Interception point: reverberant residual, still in STFT domain
    R_stft = Y_wpe - D_stft

    # 6. Back to (num_mics, F, T) and to power
    D_stft = np.transpose(D_stft, (1, 0, 2))
    R_stft = np.transpose(R_stft, (1, 0, 2))
    # P_D = np.abs(D_stft) ** 2
    # P_R = np.abs(R_stft) ** 2

    return D_stft, R_stft


def sph2cart(azimuth_deg, elevation_deg, radius):
    """
    Convert (azimuth_deg, elevation_deg, radius) -> Cartesian (x, y, z).
    Inverse of cart2sph below - same convention: elevation measured from
    the horizontal plane, azimuth measured counterclockwise from +x.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    
    return radius * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])