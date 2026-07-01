# ISIC 2018 — Skin Lesion Segmentation

This directory contains the preprocessing and training notebooks for the ISIC 2018 dataset using the UNeXt model.

## Note on Reproducibility

Part of the original dataset split was successfully recovered following a computer hardware failure. However, the model still had to be retrained. As a result, the probability maps and derived results presented in this repository may differ from those reported in the published article. These differences do not affect the main findings or conclusions of the study.

## Usage

Run the notebooks in the following order:

1. `01_preprocessing_ISIC.ipynb` — Preprocesses the ISIC 2018 dataset.
2. `02_train_inference_UNeXt_ISIC.ipynb` — Trains the UNeXt model and runs inference, saving `.npz` probability map files into `medical-imaging/data/`.
