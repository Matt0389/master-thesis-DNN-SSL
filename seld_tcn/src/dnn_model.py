
import numpy as np
import torch.nn as nn
import torch.functional as F
from pytorch_tcn import TemporalConv1d

class SeldTCN(nn.Module):

    def __init__(self):

        super().__init__()

        # Initialize Layers

            # Convs

        self.conv_block1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d()
        self.mpool1 = nn.MaxPool2d()

        self.conv_block2 = nn.Conv2d()
        self.batchnorm2 = nn.BatchNorm2d()
        self.mpool2 = nn.MaxPool2d()

        self.conv_block3 = nn.Conv2d()
        self.batchnorm3 = nn.BatchNorm2d()
        self.mpool3 = nn.MaxPool2d()

            # FCs

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()
        self.fc4 = nn.Linear()


            # TCN
            ## Every dilated conv layer needs its own FIFO buffer with a size corresponding to double the dilation factor. Relatively cheap way to get a long receptive field

        

        

    def forward(self, X):

        # Pass through conv layers

        # Pass through recurrent layers

        # Flatten the data to fit FC layers

        # Pass through FC layers

        # Return regression result, not softmax (only for classification)

        return X