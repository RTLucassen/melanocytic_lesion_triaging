# Artificial Intelligence-Based Triaging of Cutaneous Melanocytic Lesions
This repository contains all code and trained model parameters to support the paper:  

***"Artificial Intelligence-Based Triaging of Cutaneous Melanocytic lesions"***  

published in npj biomedical innovations and presented at ECDP 2025.

[[`arXiv`](https://arxiv.org/abs/2410.10509)] [[`npj BI`](https://www.nature.com/articles/s44385-025-00013-1#citeas)] [[`Poster`](https://github.com/RTLucassen/melanocytic_lesion_triaging/blob/main/.github/ECDP_poster.pdf)]

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

## Citing
If you found our work useful in your research, please consider citing our paper:
```
@article{lucassen2025artificial,
  title={Artificial intelligence-based triaging of cutaneous melanocytic lesions},
  author={Lucassen, Ruben T and Stathonikos, Nikolas and Breimer, Gerben E and Veta, Mitko and Blokx, Willeke A M},
  journal={npj Biomedical Innovations},
  volume={2},
  number={1},
  pages={10},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
