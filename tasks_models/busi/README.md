# BUSI — Breast Ultrasound Image Segmentation

This directory contains the preprocessing and training notebooks for the BUSI dataset using the UNeXt model.

## Note on Reproducibility

The original dataset split was lost due to a computer hardware failure. As a result, a new split was generated and the model had to be retrained. Consequently, the probability maps and derived results presented in this repository may differ from those reported in the published article. These differences do not affect the main findings or conclusions of the study.

## Usage

Run the notebooks in the following order:

1. `00_setup_env_UNeXt.ipynb` — Sets up the UNeXt environment.
2. `01_preprocessing_BUSI.ipynb` — Preprocesses the BUSI dataset.
3. `02_train_inference_UNeXt_BUSI.ipynb` — Trains the UNeXt model and runs inference, saving `.npz` probability map files into `medical-imaging/data/`.
