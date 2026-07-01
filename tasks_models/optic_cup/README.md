# Optic Cup — Optic Cup Segmentation

This directory contains the preprocessing and training notebooks for the optic cup segmentation task (REFUGE / ORIGA / G1020 datasets) using the SegTran model.

## Note on Reproducibility

The original dataset split was lost due to a computer hardware failure. As a result, a new split was generated and the model had to be retrained. Consequently, the probability maps and derived results presented in this repository may differ from those reported in the published article. These differences do not affect the main findings or conclusions of the study.

In addition, the original preprocessing model, `MNet_DeepCDR`, was updated to a newer version. This update enabled the extraction of three additional images containing the optic cup from the out-of-distribution dataset, which were not present in the original experimental setup.

## Usage

Run the notebooks in the following order:

1. `00_setup_env_SegTran.ipynb` — Sets up the SegTran environment.
2. `01_preprocessing_OpticCup_REFUGE.ipynb` — Preprocesses the optic cup datasets using `MNet_DeepCDR`.
3. `02_train_inference_SegTran_OpticCup.ipynb` — Trains the SegTran model and runs inference, saving `.npz` probability map files into `medical-imaging/data/`.
