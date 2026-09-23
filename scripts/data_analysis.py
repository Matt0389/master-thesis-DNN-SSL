# ==============================================================
# Script for analysing data, mainly the STARSS23 dataset,
# in order to generate synthetic data with similar distance,
#  azimuth and elevation distributions
# ==============================================================

#%%
import numpy as np
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd

file_paths = glob.glob("../data/STARSS23/MIC/metadata_dev/dev-train-**/*.csv", recursive=True)

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

plt.figure()
plt.hist(data_f["azimuth"], bins=128)
plt.title(f"Azimuth distribution for all frames of class female")
plt.xlabel("Azimuth [deg]")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(data_f["elevation"], bins=128)
plt.title(f"Elevation distribution for all frames of class female")
plt.xlabel("Elevation [deg]")
plt.ylabel("Count")
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

plt.figure()
plt.hist(data_m["azimuth"], bins=128)
plt.title(f"Azimuth distribution for all frames of class male")
plt.xlabel("Azimuth [deg]")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(data_m["elevation"], bins=128)
plt.title(f"Elevation distribution for all frames of class male")
plt.xlabel("Elevation [deg]")
plt.ylabel("Count")
plt.show()


print(f"Male frames: {cnt_male_frames}, female frames: {cnt_female_frames}")



# %%
