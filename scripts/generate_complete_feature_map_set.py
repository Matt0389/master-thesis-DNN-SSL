import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio_tools import load_feature_maps
from src.extract_salsa_feats import get_salsa_dlite, _get_salsa

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf



#================
#   Main
#================



if __name__ == "__main__":

    # ============== extract target segment =====================

    #           ========== Real data ===========
    data_path = "data/STARSS23/MIC/mic_dev/dev-train-tau/"
    annotation_path = "data/STARSS23/MIC/tau_train_female_speech_segments.csv"
    raw_anno_source_path = "data/STARSS23/MIC/metadata_dev/dev-train-tau/"

    FEATURE_DIR = Path("data/STARSS23/features")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    #           ========== Synthetic data ===========
    # data_path = "data/synthetic_starss_dataset_100_samples/mic_dev/synth_dev/"
    # annotation_path = "data/synthetic_starss_dataset_100_samples/synth100_combinedMF_speech_segments.csv"

    FS=16000    ## Change to 16000 in line with the synthetic dataset
                ## Using fs=16kHz and a HOP_SIZE=200 fits 8 columns of T per annotation, or 100ms time step 
    Ts_PER_ANNOTATION = 8
    frame_hop_sec = 0.1

    # PLaceholder array for the array of manifest rows, containing further metadata used for training
    manifest_rows = []

    # Annotation file containing the target speech clips with start frame index and segment length
    segment_df = pd.read_csv(annotation_path)

    cnt = 0
    for idx, segment in segment_df.iterrows():

        # Fname for the target .wav file
        fname = str(data_path + Path(segment["source_file"]).stem + ".wav")

        # Frame-by-frame annotation for the clip in question
        file_anno_df = pd.read_csv(raw_anno_source_path + segment["source_file"], header=None, names=["frame_number","active_class_index","source_number_index","azimuth","elevation","distance"])
                # Make this only grab the relevant frames


        # Definition of start and end frames
        start_sample = int(segment["frame_number"] * frame_hop_sec * FS)
        end_sample = int((segment["frame_number"] + segment["segment_length"]) * frame_hop_sec * FS)

        # Reading audio data using soundfile. Gets 4 channel audio data with the shape (samples, channels) and audio_len = number of frames in the clip (each frame is 100 ms)        
        audio_data, _ = sf.read(fname, start=start_sample, frames=end_sample - start_sample, always_2d=True, dtype="float32") 
        audio_data = audio_data.T
        
        print(f"Loaded data. Audio length: {segment["segment_length"]}")


        # Calculate the feature maps for this clip (C, T, F)
        features = get_salsa_dlite(audio_data)
        t_feat = features.shape[1] # number of time bins T

        print("Calculated features")

        # Store the relevant GT labels as a data frame
        anno_labels = file_anno_df.loc[segment["frame_number"] : segment["frame_number"] + segment["segment_length"] - 1, ["azimuth", "elevation", "distance"]].to_numpy()
        labels = np.repeat(anno_labels, repeats=Ts_PER_ANNOTATION, axis=0) # Repeats the GT labels such that every
        if len(labels) < t_feat:
            labels = np.pad(labels, ((0, t_feat - len(labels)), (0, 0)))
        lables = labels[:t_feat].astype(np.float32)

        # Export actual data to npz
        clip_id = f"{Path(segment['source_file']).stem}_{segment['frame_number']}"
        out_path = FEATURE_DIR / f"{clip_id}.npz"
        np.savez(out_path, features=features, labels=labels)

        # Export manifest
        manifest_rows.append({"clip_id": clip_id, "n_time_frames": t_feat, "segment_length (TEST)": segment["segment_length"]})

        cnt += 1

        print("Exported npz file")
        print(cnt)

        if cnt == 20:
            break


    pd.DataFrame(manifest_rows).to_csv(FEATURE_DIR / "manifest.csv", index=False)
    
    print(np.__version__, librosa.__version__)

    # ============== TEST RESULTING FILES WITH A LOADED SAMPLE MAP
    npz_test = pd.read_csv(FEATURE_DIR+ "/manifest.csv")
    fname_test = npz_test["clip_id"] + ".npz"
    test_map = load_feature_maps(fname)
    print(test_map)