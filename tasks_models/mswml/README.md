# MSWML — Multiple Sclerosis White Matter Lesion Segmentation

This directory contains the inference notebooks for the MSWML dataset (Shifts Challenge) using the MSWML model.

## Note on Reproducibility

The dataset splits and model weights for this task were provided by the original authors and were therefore preserved intact. The results presented in this repository are expected to be fully consistent with those reported in the published article.

## Usage

Run the notebooks in the following order:

1. `installation_repo.ipynb` — Installs the MSWML repository and its dependencies.
2. `inference.ipynb` — Runs inference using the provided model weights.
3. `create_pickle_files.ipynb` — Converts the inference outputs into `.npz` probability map files for use in the downstream evaluation notebooks.
