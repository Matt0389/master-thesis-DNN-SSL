# =============================================================================
# ========== Tools and utilities for DNN intialization and training ===========
# =============================================================================

import numpy as np
import torch.nn as nn
import torch.functional as F
import torchaudio
from torch.utils.data import Dataset


class SSL_Dataset(Dataset):

    def __init__(self):

        super.__init__()

        # xy = imported file

        # Convert x and y to torch: x = torch.from_numpy()

        # self.n_samples = xy.shape[0]


        # Determine number of time bins to use for training here?


    def __getitem__(self, index):

        return (self.x[index], self.y[index])


    def __len__(self):

        return self.n_samples



class SeldNET_causal_GPU(nn.Module):

    def __init__(self):

        super().__init__()

        # Initialize Layers

            ## Conv layers

            ## Recurrent Layers

            ## FC layers


    def forward(self, X):

        # Pass through conv layers

        # Pass through recurrent layers

        # Flatten the data to fit FC layers

        # Pass through FC layers

        # Return regression result, not softmax (only for classification)

        return X


class SeldNET_causal_MCU(nn.Module):

    def __init__(self):

        super().__init__()

        # Initialize Layers

            ## Conv layers

            ## Recurrent Layers

            ## FC layers


    def forward(self, X):

        # Pass through conv layers

        # Pass through recurrent layers

        # Flatten the data to fit FC layers

        # Pass through FC layers

        # Return regression result, not softmax (only for classification)

        return X