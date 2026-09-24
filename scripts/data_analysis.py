# ==============================================================
# Script for analysing data, mainly the STARSS23 dataset,
# in order to generate synthetic data with similar distance,
#  azimuth and elevation distributions
# ==============================================================

#%%
from pathlib import Path
import numpy as np
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd

# =============================== ANNOTATION ANALYSIS =======================================
#
# Analysis of the distributions of azimuth, elevation and distance across male and female speaker frames

# file_paths = glob.glob("../data/STARSS23/MIC/metadata_dev/dev-train-**/*.csv", recursive=True) # STARSS23 DATA

file_paths = glob.glob("../data/synthetic_starss_dataset_100_samples/metadata_dev/synth_dev/*.csv", recursive=True) # SYNTH DATA

columns = ["frame", "class", "source", "azimuth", "elevation", "distance"]
target_cols = ["class", "azimuth", "elevation", "distance"]

data_f = {col: [] for col in target_cols}


#%% ======= Female class ==========
for file_path in file_paths:
    with open(file_path) as f:

        df = pd.read_csv(f, header=None, names=columns)

        matches = df[df["class"] == 0] # 0 = Female class

        for col in target_cols:
    
            data_f[col].extend(matches[col].tolist())


cnt_female_frames = len(data_f["class"])


plt.figure()
plt.hist(data_f["distance"], bins=128)
plt.title(f"Distance distribution for all frames of class female")
plt.xlabel("Distance [cm]")
plt.ylabel("Count")
plt.show()

# plt.figure()
# plt.hist(data_f["azimuth"], bins=128)
# plt.title(f"Azimuth distribution for all frames of class female")
# plt.xlabel("Azimuth [deg]")
# plt.ylabel("Count")
# plt.show()

# plt.figure()
# plt.hist(data_f["elevation"], bins=128)
# plt.title(f"Elevation distribution for all frames of class female")
# plt.xlabel("Elevation [deg]")
# plt.ylabel("Count")
# plt.show()

#%% ============= Cool seaborn plot =========

import seaborn as sns

# assuming you already have this from before:
# data = {"azimuth": [...], "elevation": [...], "distance": [...]}

g = sns.jointplot(
    x=data_f["azimuth"],
    y=data_f["elevation"],
    kind="scatter",
    marginal_kws=dict(bins=128),
    height=6,
    s=5,          # marker size, small since you likely have many points
    alpha=0.4     # transparency helps when points overlap heavily
)

g.set_axis_labels("Degrees", "Degrees")
g.ax_joint.set_xlabel("Azimuth (Degrees)")
g.ax_joint.set_ylabel("Elevation (Degrees)")
g.fig.suptitle("Female Speech", y=1.02)

# plt.savefig("azimuth_elevation_joint.png", dpi=150, bbox_inches="tight")
plt.show()




#%% ======= Male class ==========

data_m = {col: [] for col in target_cols}

for file_path in file_paths:
    with open(file_path) as f:

        df = pd.read_csv(f, header=None, names=columns)

        matches = df[df["class"] == 1] # 1 = Male class

        for col in target_cols:
    
            data_m[col].extend(matches[col].tolist())


cnt_male_frames = len(data_m["class"])

plt.figure()
plt.hist(data_m["distance"], bins=128)
plt.title(f"Distance distribution for all frames of class male")
plt.xlabel("Distance [cm]")
plt.ylabel("Count")
plt.show()

# plt.figure()
# plt.hist(data_m["azimuth"], bins=128)
# plt.title(f"Azimuth distribution for all frames of class male")
# plt.xlabel("Azimuth [deg]")
# plt.ylabel("Count")
# plt.show()

# plt.figure()
# plt.hist(data_m["elevation"], bins=128)
# plt.title(f"Elevation distribution for all frames of class male")
# plt.xlabel("Elevation [deg]")
# plt.ylabel("Count")
# plt.show()


print(f"Male frames: {cnt_male_frames}, female frames: {cnt_female_frames}")



# %%

g = sns.jointplot(
    x=data_m["azimuth"],
    y=data_m["elevation"],
    kind="scatter",
    marginal_kws=dict(bins=128),
    height=6,
    s=5,          # marker size, small since you likely have many points
    alpha=0.4     # transparency helps when points overlap heavily
)

g.set_axis_labels("Degrees", "Degrees")
g.ax_joint.set_xlabel("Azimuth (Degrees)")
g.ax_joint.set_ylabel("Elevation (Degrees)")
g.fig.suptitle("Male Speech", y=1.02)

# plt.savefig("azimuth_elevation_joint.png", dpi=150, bbox_inches="tight")
plt.show()








#%% ============================= JITTER ANALYSIS =======================================
#  
#   Examine the jitter of azimuth, elevation and distance across STARSS23 annotations.abs
#   Used for generating a more representative synthetic dataset


def circular_diff(a, b, period=360.0):
    """Shortest signed angular distance from b to a, wrapping through +/-period/2."""
    return (a - b + period / 2) % period - period / 2

def measure_annotation_jitter(folder, classes, max_gap_frames=1):
    """
    Scan a folder of STARSS23 annotation CSVs and measure how much
    azimuth/elevation/distance vary frame-to-frame WITHIN a single
    continuous event (same file, class, and source_number_index, with
    frame numbers no more than max_gap_frames apart). Azimuth uses a
    wraparound-aware circular difference (it's a circular quantity,
    unlike elevation and distance) so a crossing near +/-180deg doesn't
    register as a huge spurious jump.
    """
    az_diffs, el_diffs, dist_diffs = [], [], []

    for csv_path in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(csv_path, header=None,
                          names=["frame_number", "active_class_index", "source_number_index",
                                 "azimuth", "elevation", "distance"])
        df = df[df["active_class_index"].isin(classes)]

        for _, group in df.groupby(["active_class_index", "source_number_index"]):
            group = group.sort_values("frame_number")
            frames = group["frame_number"].to_numpy()
            if len(frames) < 2:
                continue
            az = group["azimuth"].to_numpy()
            el = group["elevation"].to_numpy()
            dist = group["distance"].to_numpy()

            gaps = np.diff(frames)
            consecutive = gaps <= max_gap_frames

            az_step = np.abs(circular_diff(az[1:], az[:-1]))
            az_diffs.extend(az_step[consecutive])
            el_diffs.extend(np.abs(np.diff(el))[consecutive])
            dist_diffs.extend(np.abs(np.diff(dist))[consecutive])

    def summarize(x, label):
        x = np.array(x)
        if len(x) == 0:
            print(f"{label}: no data found")
            return {}
        stats = {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()),
                 "p90": float(np.percentile(x, 90)), "n": len(x)}
        print(f"{label}: mean={stats['mean']:.2f}  median={stats['median']:.2f}  "
              f"std={stats['std']:.2f}  p90={stats['p90']:.2f}  (n={stats['n']})")
        return stats

    return {
        "azimuth_deg": summarize(az_diffs, "Azimuth frame-to-frame |diff| (deg)"),
        "elevation_deg": summarize(el_diffs, "Elevation frame-to-frame |diff| (deg)"),
        "distance_cm": summarize(dist_diffs, "Distance frame-to-frame |diff| (cm)"),
    }

print("\n================== FOR dev-train-tau ==================")
x = measure_annotation_jitter(folder="/mnt/88887733-0633-4959-9114-d6be9eff8a9f/Thesis/master-thesis-DNN-SSL/data/STARSS23/MIC/metadata_dev/dev-train-tau", classes=[0, 1])

print("\n================== FOR dev-test-tau ==================")
x = measure_annotation_jitter(folder="/mnt/88887733-0633-4959-9114-d6be9eff8a9f/Thesis/master-thesis-DNN-SSL/data/STARSS23/MIC/metadata_dev/dev-test-tau", classes=[0, 1])

print("\n================== FOR dev-train-sony ==================")
x = measure_annotation_jitter(folder="/mnt/88887733-0633-4959-9114-d6be9eff8a9f/Thesis/master-thesis-DNN-SSL/data/STARSS23/MIC/metadata_dev/dev-train-sony", classes=[0, 1])

print("\n================== FOR dev-test-sony ==================")
x = measure_annotation_jitter(folder="/mnt/88887733-0633-4959-9114-d6be9eff8a9f/Thesis/master-thesis-DNN-SSL/data/STARSS23/MIC/metadata_dev/dev-test-sony", classes=[0, 1])

print("\n================== FOR synth-dev ==================")
x = measure_annotation_jitter(folder="/mnt/88887733-0633-4959-9114-d6be9eff8a9f/Thesis/master-thesis-DNN-SSL/data/synthetic_starss_dataset_100_samples/metadata_dev/synth_dev", classes=[0, 1])

# %%
