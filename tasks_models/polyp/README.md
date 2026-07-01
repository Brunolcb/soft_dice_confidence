# Polyp — Polyp Segmentation

This directory contains the inference notebooks for the polyp segmentation task (Kvasir-SEG / CVC datasets/ ETIS dataset) using the Polyp-PVT model.

## Note on Reproducibility

The original dataset splits provided by the authors were preserved. However, the original probability maps were no longer available following a computer hardware failure. Therefore, they were regenerated using the model weights provided by the authors. The results presented in this repository may exhibit minor differences from those reported in the published article, though these differences do not affect the main findings or conclusions of the study.

## Usage

Run the notebooks in the following order:

1. `00_register_polyp_project_kernel.ipynb` — Registers the Polyp-PVT project kernel.
2. `01_inference_PolypPVT.ipynb` — Runs inference using the provided model weights and saves `.npz` probability map files into `medical-imaging/data/`.
