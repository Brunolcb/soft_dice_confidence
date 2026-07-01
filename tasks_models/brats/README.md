# BraTS — Brain Tumour Segmentation

This directory contains the post-processing notebook for the BraTS 2021 dataset using nnU-Net predictions.

## Note on Reproducibility

The original dataset split was lost due to a computer hardware failure. As a result, a new split was generated and the model had to be retrained. Consequently, the probability maps and derived results presented in this repository may differ from those reported in the published article. These differences do not affect the main findings or conclusions of the study.

## Usage

Run the following notebook to generate the probability maps from the nnU-Net outputs:

1. `brats_post_processing.ipynb` — Post-processes nnU-Net predictions and saves `.npz` probability map files into `medical-imaging/data/`.
