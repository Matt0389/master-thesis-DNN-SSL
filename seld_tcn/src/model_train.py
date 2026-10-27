
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dnn_data import SSL_Dataset


feature_dir = Path(__file__).resolve().parent.parent / "features"
label_dir = Path(__file__).resolve().parent.parent.parent / "data" / "STARSS23" / "MIC" / "metadata_dev" / "dev-train-tau"

print("feature_dir:", feature_dir)
print("  exists:", feature_dir.exists())

print("label_dir:", label_dir)
print("  exists:", label_dir.exists())

dataset = SSL_Dataset(
        feature_dir = feature_dir,   
        label_dir = label_dir,      
        chunk_seconds = 8,
    )

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

x, y, mask = next(iter(loader))

print("batch shapes:", x.shape, y.shape, mask.shape)
