# Master-Thesis-DNN-SSL

Repository for my master thesis in Engineering Acoustics at DTU, titled "Design of deep neural networks for sound source localization under hardware constraints".

## Project brief

This project aims to explore DNN-based solutions to estimating the direction-of-arrival (DOA) and distance of sources in a monophonic environment. In the project, the relevant cues will be identified and extracted using classical DSP methods, after which the feature maps will be input into the various network types for training and inference. As part of the exploration, experiments will be conducted using several promising DNN architectures within the broader field of SSL, such as CNNs, CRNNs, TCNs, and attention-based architectures. \\

Models will initially be trained in PyTorch and evaluated on GPU to serve as a baseline of the models' individual capability. Afterwards, extensive effort will be expended to optimize these models for deployment on an MCU, namely the STM32 N6. Here, the constrained models will be evaluated on their SSL error rate, as well as inference latency and MACs. \\

Finally, the constrained models will be evaluated against each other, based on error rates, MACs, latency, as well as how well optimized the model architecture is for accomodating the important acoustic cues.

## Acoustic cues and features

## Datasets

## Architecture choices

## Hardware and -constraints

## Constrained model alterations
