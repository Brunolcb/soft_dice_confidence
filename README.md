# Soft Dice Confidence

This repository contains the code and experiments for the paper:

> **Selective Segmentation with Soft Dice Confidence**

Soft Dice Confidence (SDC) is a confidence estimator designed for selective prediction at the image level in semantic segmentation tasks. It ranks segmentation predictions by their expected Dice performance, enabling a model to abstain from low-confidence predictions and improve reliability.

---

## Repository Structure

```
soft_dice_confidence/
├── sdc_env_creation.ipynb               # Environment setup — run this first
├── environment.yml                      # Conda environment specification
├── synthetic-selective-segmentation/    # Synthetic data experiments
└── medical-imaging/                     # Real medical imaging experiments
    ├── tasks_models/                    # Per-dataset model training & inference
    │   ├── busi/                        # BUSI — UNeXt
    │   ├── isic/                        # ISIC 2018 — UNeXt
    │   ├── polyp/                       # Kvasir/CVC — Polyp-PVT
    │   ├── optic_cup/                   # REFUGE/ORIGA/G1020 — SegTran
    │   ├── brats/                       # BraTS — post-processing
    │   └── mswml/                       # MSWML (MS lesion) — Shifts challenge
    ├── RC_curves.ipynb                  # Risk–Coverage curves (ID)
    ├── RC_curves_OOD.ipynb              # Risk–Coverage curves (OOD)
    ├── Optimum_uncertainty_thresholds.ipynb  # AEF uncertainty threshold search
    ├── Features_creation.ipynb          # AEF feature extraction
    ├── AURCxtuning_size_curves.ipynb    # AURC vs tuning size (ID)
    ├── AURCxtuning_size_curves_OOD.ipynb    # AURC vs tuning size (OOD)
    ├── Coveragextuning_size_curves-OOD.ipynb # Coverage vs tuning size (OOD)
    ├── bounds_datasets.ipynb            # Theoretical bounds (ID)
    └── bounds_datasets_OOD.ipynb       # Theoretical bounds (OOD)
```

---

## Environment Setup

Create the conda environment from the specification file, or follow the guided setup notebook:

```bash
conda env create -f environment.yml
conda activate sdc_env
```

Alternatively, open and run `sdc_env_creation.ipynb`, which walks through the full installation and kernel registration needed to reproduce all notebooks.

---

## Synthetic Experiments

The `synthetic-selective-segmentation/` folder contains self-contained experiments on synthetic data that validate the theoretical properties of SDC under controlled generative models. The experiments explore scenarios with known and approximated output probability distributions, custom pixel-label distributions, and a downsampled version of the ISIC dataset.

| Notebook | Description |
|---|---|
| `exp1-marginals-and-known-probs.ipynb` | Uniform X, beta pixel labels, known probabilities |
| `exp2-marginals-and-approx-probs.ipynb` | Same setup, approximated probabilities |
| `exp3-custom-distribution-known-probs.ipynb` | Custom pixel distribution, known probabilities |
| `exp4-custom-distribution-approx-probs.ipynb` | Custom pixel distribution, approximated probabilities |
| `exp5-downsampled-ISIC.ipynb` | SDC on a downsampled real-world ISIC subset |
| `paper-plots.ipynb` | Generates the synthetic figures used in the paper |

---

## Medical Imaging Experiments

The medical imaging experiments follow a fixed pipeline. Run the steps below **in order**.

### Step 1 — Per-dataset Model Training & Inference (`tasks_models/`)

Each sub-folder trains the task model, runs inference, and saves probability maps (`.npz` files with keys `p_hat` and `y`) into `medical-imaging/data/`. Run the preprocessing and inference notebooks for every dataset you want to reproduce:

| Dataset | Task | Model | Notebooks |
|---|---|---|---|
| **BUSI** | Breast ultrasound lesion segmentation | UNeXt | `00_setup_env_UNeXt.ipynb` → `01_preprocessing_BUSI.ipynb` → `02_train_inference_UNeXt_BUSI.ipynb` |
| **ISIC 2018** | Skin lesion segmentation | UNeXt | `01_preprocessing_ISIC.ipynb` → `02_train_inference_UNeXt_ISIC.ipynb` |
| **Kvasir / CVC (Polyp)** | Polyp segmentation | Polyp-PVT | `00_register_polyp_project_kernel.ipynb` → `01_inference_PolypPVT.ipynb` |
| **REFUGE / ORIGA / G1020 (Optic Cup)** | Optic cup segmentation | SegTran | `00_setup_env_SegTran.ipynb` → `01_preprocessing_OpticCup_REFUGE.ipynb` → `02_train_inference_SegTran_OpticCup.ipynb` |
| **BraTS** | Brain tumour segmentation | MSWML ensemble | `brats_post_processing.ipynb` |
| **MSWML (MS lesion)** | Multiple sclerosis white matter lesion segmentation | MSWML (Shifts challenge) | `installation_repo.ipynb` → `inference.ipynb` → `create_pickle_files.ipynb` |

### Step 2 — Risk–Coverage Curves

```
medical-imaging/RC_curves.ipynb        # In-distribution (ID) results
medical-imaging/RC_curves_OOD.ipynb   # Out-of-distribution (OOD) results
```

These notebooks load the `.npz` probability files, compute Risk–Coverage (RC) curves for each confidence estimator, and generate the RC curve figures used in the paper.

### Step 3 — AEF Uncertainty Thresholds

```
medical-imaging/Optimum_uncertainty_thresholds.ipynb
```

Searches for the optimal per-estimator uncertainty thresholds on the tuning sets. The output is saved as a pickle file consumed by the next step.

### Step 4 — AEF Feature Extraction

```
medical-imaging/Features_creation.ipynb
```

Builds the feature matrix used to train the Random Forest for the Automatic Estimator Fusion (AEF) confidence estimator. Must be run after `Optimum_uncertainty_thresholds.ipynb`.

### Step 5 — AURC vs Tuning Size Curves

```
medical-imaging/AURCxtuning_size_curves.ipynb        # ID
medical-imaging/AURCxtuning_size_curves_OOD.ipynb   # OOD
```

Generates the curves showing how the AURC of each estimator varies with the size of the tuning set.

### Step 6 — Coverage vs Tuning Size (OOD)

```
medical-imaging/Coveragextuning_size_curves-OOD.ipynb
```

Generates the coverage-versus-tuning-size curves for the out-of-distribution setting.

### Step 7 — Theoretical Bounds

```
medical-imaging/bounds_datasets.ipynb        # ID bounds
medical-imaging/bounds_datasets_OOD.ipynb   # OOD bounds
```

Computes and visualises the theoretical performance bounds for the selective segmentation framework on each dataset.

---

## Datasets

| Dataset | Task | Access |
|---|---|---|
| **BUSI** — Breast Ultrasound Images Dataset | Breast lesion segmentation | [Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) · [Paper](https://doi.org/10.1016/j.dib.2019.104863) |
| **ISIC 2018** — Skin Lesion Analysis | Skin lesion segmentation | [ISIC Archive](https://challenge.isic-archive.com/landing/2018/) · [Paper](https://doi.org/10.48550/arXiv.1902.03368) |
| **Kvasir-SEG** | Polyp segmentation (ID) | [Simula](https://datasets.simula.no/kvasir-seg/) · [Paper](https://doi.org/10.1007/978-3-030-37734-2_37) |
| **CVC-ClinicDB** | Polyp segmentation (OOD) | [CVC](https://polyp.grand-challenge.org/CVCClinicDB/) · [Paper](https://doi.org/10.1016/j.compmedimag.2015.02.007) |
| **CVC-300 / CVC-ColonDB / ETIS-LaribPolypDB** | Polyp segmentation (OOD) | [Polyp-PVT repo](https://github.com/DengPingFan/Polyp-PVT) |
| **REFUGE** | Optic cup segmentation (ID) | [Grand Challenge](https://refuge.grand-challenge.org/) · [Paper](https://doi.org/10.1016/j.media.2019.101570) |
| **ORIGA** | Optic cup segmentation (OOD) | [SERI](https://www.seri.com.sg/) · [Paper](https://doi.org/10.1167/iovs.10-6986) |
| **G1020** | Optic cup segmentation (OOD) | [G1020](https://arxiv.org/abs/2006.09158) |
| **BraTS 2021** | Brain tumour segmentation | [Synapse](https://www.synapse.org/#!Synapse:syn27046444/wiki/616992) · [Paper](https://doi.org/10.1109/TMI.2014.2377694) |
| **MSWML / Shifts MS** | MS white matter lesion segmentation | [Zenodo part 1](https://zenodo.org/records/7051658) · [Zenodo part 2](https://zenodo.org/records/7051692) · [Paper](https://doi.org/10.48550/arXiv.2206.08086) |

---

## Model Repositories

| Model | Reference |
|---|---|
| **UNeXt** | [jeya-maria-jose/UNeXt-pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch) · [Paper](https://doi.org/10.48550/arXiv.2203.04967) |
| **Polyp-PVT** | [DengPingFan/Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) · [Paper](https://doi.org/10.48550/arXiv.2108.06932) |
| **SegTran** | [jonfan/segtran](https://github.com/askerlee/segtran) · [Paper](https://doi.org/10.48550/arXiv.2102.07016) |
| **MSWML** | [Shifts Challenge](https://github.com/Shifts-Project/shifts) · [Paper](https://doi.org/10.48550/arXiv.2206.08086) |
| **MNet DeepCDR** (optic disc/cup pre-processing) | [HzFu/MNet_DeepCDR](https://github.com/HzFu/MNet_DeepCDR) · [Paper](https://doi.org/10.1109/TMI.2018.2885446) |

---

## Citation

If you use this code or the SDC estimator in your work, please cite:

```bibtex
@misc{borges2026softdiceconfidencenearoptimal,
      title={Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation}, 
      author={Bruno Laboissiere Camargos Borges and Bruno Machado Pacheco and Danilo Silva},
      year={2026},
      eprint={2402.10665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.10665}, 
}
```
