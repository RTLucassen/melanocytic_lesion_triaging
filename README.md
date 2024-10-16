# Artificial Intelligence-Based Triaging of Cutaneous Melanocytic Lesions
This repository contains all code and trained model parameters to support the paper:  

***"Artificial Intelligence-Based Triaging of Cutaneous Melanocytic lesions"***  

which is currently under review.

[[`arXiv`](https://arxiv.org/abs/2410.10509)]

<div align="center">
  <img width="100%" alt="Method" src=".github\method.png">
</div>

## Contents
The repository contains several folders:
- `configs` contains two files that include the configurations used for data preprocessing and model training.
- `evaluation` contains all python files that were used for evaluation of individual models and the ensemble. 
- `HIPT` contains the model implementation and trained parameters for the last ViT in HIPT. 
Pretrained parameters for the first two ViTs in HIPT can be downloaded from the original [repository](https://github.com/mahmoodlab/HIPT).
- `pipeline` contains all python files that were used for data transfer, de-identification, tissue segmentation, tessellation, and feature extraction. 
The implementation was designed to perform the preprocessing tasks in parallel, 
which may limit the generalizibility of the pipeline to different infrastructure and data storage systems.
- `simulation_study` contains the implementation of the simulation experiment.
- `start` contains all files to start the preprocessing tasks.
- `training` contains the implementation of the model training loop.
