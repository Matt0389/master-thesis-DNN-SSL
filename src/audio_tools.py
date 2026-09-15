import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

# STARSS23 annotations are given at a 10 Hz frame rate -> 100 ms per frame.
FRAME_HOP_SEC = 0.1


def frame_extract(fname, frame_idx, frame_hop_sec=FRAME_HOP_SEC, audio_dir=None):
    """
    Extract the audio segment corresponding to a single annotation frame.

    Parameters
    ----------
    fname : str or Path
        Filename (or full path) of the recording this frame belongs to.
        If it ends in '.csv' (e.g. taken straight from a 'source_file'
        column), the extension is swapped for '.wav' automatically.
    frame_idx : int
        Frame index as given in the STARSS23 annotation (0-based).
    frame_hop_sec : float
        Duration of one frame in seconds. STARSS23 labels run at 10 Hz,
        i.e. 100 ms per frame. Change this if your annotations differ.
    audio_dir : str or Path, optional
        Directory containing the wav files. If given, `fname` is treated
        as just a filename and joined with this directory.

    Returns
    -------
    audio : np.ndarray
        Shape (num_samples, num_channels) for multichannel audio
        (FOA/MIC formats both have >1 channel in STARSS23).
    sr : int
        Sample rate of the file.
    """
    fname = Path(fname)
    if audio_dir is not None:
        fname = Path(audio_dir) / fname.name
    fname = fname.with_suffix(".wav")

    info = sf.info(fname)
    sr = info.samplerate

    start_sample = int(round(frame_idx * frame_hop_sec * sr))
    num_samples = int(round(frame_hop_sec * sr))

    # Guard against a frame index that runs past the end of the file
    start_sample = min(start_sample, info.frames)
    num_samples = min(num_samples, max(info.frames - start_sample, 0))

    audio, sr = sf.read(fname, start=start_sample, frames=num_samples, always_2d=True)
    return audio, sr