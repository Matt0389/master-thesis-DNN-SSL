

#%% =========== IMPORTS ============

import os
import spatialscaper as ss
import numpy as np
import sys
import glob
import librosa
import random
sys.path.insert(0, "../SpatialScaper/scripts")

from prepare_rirs import METU_REMOTES, prepare_metu, download_and_extract_remotes, TAU_REMOTES, download_tau, prepare_tau
from pathlib import Path




#%% ================ BUILD DIRECTORY FOR LIBRISPEECH DATASET ================

def parse_librispeech_speakers(root):
    """Parse LibriSpeech's SPEAKERS.TXT into {speaker_id (str): gender ('F'/'M')}."""
    speakers = {}
    with open(Path(root) / "SPEAKERS.TXT", "r") as f:
        for line in f:
            if line.startswith(";"):
                continue
            fields = [x.strip() for x in line.split("|")]
            if len(fields) < 2:
                continue
            speaker_id, gender = fields[0], fields[1]
            speakers[speaker_id] = gender
    return speakers

def build_foreground_dir(librispeech_root, out_dir, subset="train-clean-100"):
    """
    Symlinks LibriSpeech utterances into <out_dir>/femaleSpeech/ and
    <out_dir>/maleSpeech/, the folder-per-class layout SpatialScaper's
    get_label_list/get_files_list expect (confirmed: it just globs
    whatever files sit inside each class-named subfolder).
    """
    speakers = parse_librispeech_speakers(librispeech_root)  # from our earlier pipeline
    out_dir = Path(out_dir)
    (out_dir / "femaleSpeech").mkdir(parents=True, exist_ok=True)
    (out_dir / "maleSpeech").mkdir(parents=True, exist_ok=True)

    root = Path(librispeech_root) / subset
    for speaker_dir in root.iterdir():
        if not speaker_dir.is_dir():
            continue
        gender = speakers.get(speaker_dir.name)
        if gender not in ("F", "M"):
            continue
        target_folder = "femaleSpeech" if gender == "F" else "maleSpeech"
        for flac_path in speaker_dir.rglob("*.flac"):
            link_path = out_dir / target_folder / f"{speaker_dir.name}_{flac_path.name}"
            if not link_path.exists():
                os.symlink(flac_path.resolve(), link_path)

    print(f"Linked {len(list((out_dir/'femaleSpeech').iterdir()))} female, "
          f"{len(list((out_dir/'maleSpeech').iterdir()))} male files.")


build_foreground_dir("../data/LibriSpeech", "../data/synth/foreground_librispeech")


# %% ================== DOWNLOAD RIR DATA =====================

    
source_path = Path("../data/synth/rir_datasets/source_data")
sofa_path = Path("../data/synth/rir_datasets/spatialscaper_RIRs")
sofa_path.mkdir(parents=True, exist_ok=True)

## This downloads the METU classroom IRs
# metu_path = source_path / METU_REMOTES["database_name"]
# download_and_extract_remotes(METU_REMOTES["remotes"], metu_path, cleanup=True)
# prepare_metu(metu_path, sofa_path)


## This downloads the TAU collection of RIRs
# tau_path = source_path / TAU_REMOTES["database_name"]
# download_tau(tau_path, TAU_REMOTES["remotes"], cleanup=True)
# prepare_tau(tau_path, sofa_path, formats=["mic"])   # only "mic" 




# %% ==================== INIT SIM FUNCTIONS  ======================

FOREGROUND_DIR = "../data/synth/foreground_librispeech"
RIR_DIR = "../data/synth/rir_datasets"
# ROOM = "gym"
FORMAT = "mic"
SR = 16000
OUTPUT_DIR = "../data/synth/output"
TARGET_DURATION = [16.0, 20.0]  # Keep all clips at either of these lengths, with padding
PADDING_SEC = 0.5

def get_utterance_files(foreground_dir):
    """Every source audio file across all class subfolders (femaleSpeech/, maleSpeech/)."""
    return sorted(glob.glob(os.path.join(foreground_dir, "*", "*.flac")))

# def choose_duration_and_padding(utterance_duration):
#     """
#     Decide which fixed clip length this utterance should go into, and how
#     much silence to put on each side to center it.
#     """
#     if utterance_duration < 10.0:

#         return 
    
#     if utterance_duration > TARGET_DURATION[0]:

#         total_padding = TARGET_DURATION[1] - utterance_duration

#         return (TARGET_DURATION[1], total_padding / 2)

#     else:

#         total_padding = TARGET_DURATION[0] - utterance_duration

#         return (TARGET_DURATION[0], total_padding / 2)



def generate_soundscape(index, source_file, rm):

    track_name = f"fold1_{rm}_mix{index+1:03d}"
    label = os.path.basename(os.path.dirname(source_file))  # "femaleSpeech" / "maleSpeech"
    duration = librosa.get_duration(path=source_file) + PADDING_SEC
    # duration = 18.0 # Truncating to even 16 seconds - useful for further segmentation
    # utterance_duration = librosa.get_duration(path=source_file)
    # duration, pad = choose_duration_and_padding(utterance_duration)

    ssc = ss.Scaper(
        duration,
        FOREGROUND_DIR,
        RIR_DIR,
        FORMAT,
        room=rm,
        sr=SR,
        use_room_ambient_noise=True,    # Create background noise according to room ambience
        max_event_dur=duration,          # >= full utterance length - never truncated
        speed_limit=0.2,                # m/s - small head-jitter-scale movement, not room-traversing motion
    )
    ssc.ref_db = -65

    ssc.add_background()

    ssc.add_event(
        label=("const", label),
        source_file=("const", source_file),
        source_time=("const", 0),
        event_time=("const", 0.3),       # add silence padding
        snr=("uniform", 6, 20),
    )

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)
    ssc.generate(audiofile, labelfile)

utterance_files = get_utterance_files(FOREGROUND_DIR)
rooms = ["metu", "bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"]

random.seed(43)
random.shuffle(utterance_files)








#%% ============= GENERATE AUDIO DATA WITH SPATIALSCAPER =================

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from collections import defaultdict


# ===== Patch noise broadcasting issue

db2multiplier = ss.core.db2multiplier
def _patched_get_background_noise(self, out_audio):

    """
    When generating, the length of the ambient noise is sometimes 1 off from the audio, and they can thus not be broadcast together. This function implements a patch for this issue.
    """

    target_len = out_audio.shape[0]
    for event in self.bg_events:
        if not event.source_file:
            ambient = self.generate_noise(event)
        else:
            if event.source_time is not None:
                ambient, _ = librosa.load(
                    event.source_file, sr=self.sr,
                    offset=event.source_time, duration=event.event_duration,
                )
            else:
                ambient, _ = librosa.load(event.source_file, sr=self.sr)
                repeats = -(-target_len // len(ambient))
                ambient = np.tile(ambient, repeats)[:target_len]
            ambient = ambient[:, np.newaxis]

        # length guard - fixes off-by-one regardless of source
        n = ambient.shape[0]
        if n < target_len:
            ambient = np.pad(ambient, ((0, target_len - n), (0, 0)), mode="edge")
        elif n > target_len:
            ambient = ambient[:target_len]

        scale = db2multiplier(self.ref_db + event.snr, np.mean(np.abs(ambient)))
        out_audio += scale * ambient
    return out_audio

ss.core.Scaper.get_background_noise = _patched_get_background_noise





# ================== For running parallel CPU processes

## The normal "generate_soundscape()" function seems to load the relevant IR on each call. Below I utilize the LRU cache to have each CPU worker load the relevant room into memory, and keep it there until all utterances have been processed. 

_original_get_room_irs = ss.core.Scaper.get_room_irs_wav_xyz

@lru_cache(maxsize=1)   # keep only the current room in memory per process

def _cached_room_irs(rir_dir, room, fmt, sr):
    """
    Generate a temp soundscape with the room in question, just to load the RIR
    """
    tmp = ss.Scaper(1.0, FOREGROUND_DIR, rir_dir, fmt, room=room, sr=sr)

    return _original_get_room_irs(tmp)

def _patched_get_room_irs(self, wav=True, pos=True):
    """
    Return the cached RIR
    """
    return _cached_room_irs(self.rir_dir, self.room, self.format, self.sr)


ss.core.Scaper.get_room_irs_wav_xyz = _patched_get_room_irs # Make sure the cached RIR is utilized

by_room = defaultdict(list)

recs_per_room = 5

for i, f in enumerate(utterance_files):
    by_room[rooms[i // recs_per_room % len(rooms)]].append((i, f))   # spread files across rooms, grouped


def _run_room_batch(args):
    """
    Run function for each worker
    """
    
    room, items = args

    for i, source_file in items:
        
        if i >= recs_per_room * len(rooms):
            break

        generate_soundscape(i, source_file, room)


with ProcessPoolExecutor(max_workers=2) as ex:

    list(ex.map(_run_room_batch, by_room.items()))


# %%
