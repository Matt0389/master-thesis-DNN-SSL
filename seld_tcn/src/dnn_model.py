
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    """
    Inherited from SELDnet, with structure:
    Conv2D -> Batchnorm2D -> ReLU -> MaxPool2D -> dropout
    """

    def __init__(self, in_channel, out_channel, pool = (3, 3), dropout_rate=0.5):

        super().__init__()

        # Regular 2d convolution. Padding happens in the forward step
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=0)

        # From "UNDERSTANDING DEEP LEARNING" : 
        # BN shifts shifts and rescales each activation so that its mean and variance across the batch become values that are learned during training.
        # The network becomes invariant to rescaling weights and biases, meaning a doubling in weight and bias results in a doubling in activation.
        # BN stabilizes forward propagation and can yield higher learning rates, against increasing the parameter count slightly, since bias and variance 
        # is now learned for each hidden unit. This results in redundancy but also the aforementioned benefits. Additionally, the model is better regularized due to the injection of batch-dependent noise in the process.
        self.bn = nn.BatchNorm2d(out_channel)

        # This operation pools the output of the convolution in order to downsample the activations along the TF axes, given by the "pool" tuple. The max value within the kernel is selected to represent it, and the remaining values are dropped
        self.mpool = nn.MaxPool2d(kernel_size=pool)

        # From "UNDERSTANDING DEEP LEARNING" : 
        # Dropout clamps a random subset of hidden units to zero at each iteration, making the network less dependent on any given hidden unit and encourages smaller weights. Thus the overall change in the function due to the presence or absence of any specific hidden unit is reduced.
        self.drop = nn.Dropout(dropout_rate)


    def forward(self, x):

        # This padding step takes care of the padding in the convolution, because no padding results in dropped feature columns. 
        # Padding in directions (F_left, F_right, T_left, T_right). No padding along the right temporal dimension means purely causal features. 
        x = F.pad(x, (1, 1, 2, 0))

        # Actual forward pass
        x = F.relu(self.bn(self.conv(x)))

        x = self.drop(self.mpool(x))

        return x

        


class ResBlock(nn.Module):

    """
    Block responsible for the strided and dilated convolutions in the TCN
    """

    def __init__(self):

        return



class SeldTCN_model(nn.Module):

    def __init__(self):

        super().__init__()

        # Initialize Layers

            # Convs

        # self.conv_block1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3)
        # self.batchnorm1 = nn.BatchNorm2d()
        # self.mpool1 = nn.MaxPool2d()

        # self.conv_block2 = nn.Conv2d()
        # self.batchnorm2 = nn.BatchNorm2d()
        # self.mpool2 = nn.MaxPool2d()

        # self.conv_block3 = nn.Conv2d()
        # self.batchnorm3 = nn.BatchNorm2d()
        # self.mpool3 = nn.MaxPool2d()

            # FCs

        # self.fc1 = nn.Linear()
        # self.fc2 = nn.Linear()
        # self.fc3 = nn.Linear()
        # self.fc4 = nn.Linear()


            # TCN
            ## Every dilated conv layer needs its own FIFO buffer with a size corresponding to double the dilation factor. Relatively cheap way to get a long receptive field

        

        

    def forward(self, X):

        # Pass through conv layers

        # Pass through recurrent layers

        # Flatten the data to fit FC layers

        # Pass through FC layers

        # Return regression result, not softmax (only for classification)

        return X