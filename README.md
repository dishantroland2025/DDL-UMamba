# DDL-UMamba: Deep Delta Learning for Boundary-Aware Medical Image Segmentation

Official repository for **DDL-UMamba**, which introduces a Deep Delta Learning block with channel compression (DDL-CC) into the U-Mamba segmentation backbone. DDL-CC replaces the backbone's additive state-space mixing with an exact, matrix-valued delta-rule update at every voxel, supervised toward label-derived object boundaries via a boundary-spectral loss.

This repository builds directly on the official U-Mamba codebase and the nnU-Net framework.

## Method overview

At each encoder stage, the DDL-CC block:

*   Expands each feature into a small associative state of $d_v$ slots.
*   Predicts a unit key $k$, a bounded gate $\beta \in [0, 2]$, and a value $v$ from a shared context head.
*   Applies a generalized Householder delta-rule update: $X' = X + \beta k(v - X^T k)^T = H(\beta,k)X + \beta kv^T$, where $H(\beta,k) = I - \beta kk^T$.
*   Compresses the refined state back to the original channel count via an $\ell_2$-normalized channel convolution.

The gate $\beta$ is additionally supervised by a boundary-spectral loss, which trains $\beta$ toward a morphological boundary map derived from the ground-truth labels (high on inter-class interfaces, low in homogeneous interiors), making $\beta$ an interpretable, anatomy-aligned signal without any auxiliary post-hoc attribution method.

Two model variants are provided, following U-Mamba's naming convention:

*   **DDL-UMamba_Bot** — a single DDL-CC block at the bottleneck only.
*   **DDL-UMamba_Enc** — a DDL-CC block after every encoder stage (primary model).

## Installation

**Requirements:** Ubuntu 20.04, CUDA 11.8

1.  **Create a virtual environment:**
    ```bash
    conda create -n ddlumamba python=3.10 -y
    conda activate ddlumamba
    ```
2.  **Install PyTorch 2.0.1+:**
    ```bash
    pip install torch==2.2.0 torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
3.  **Install Mamba:**
    ```bash
    pip install causal-conv1d>=1.2.0
    pip install mamba-ssm --no-cache-dir
    ```
4.  **Download this repository:**
    ```bash
    git clone [https://github.com/dishantdas2004/DDL-UMamba](https://github.com/dishantdas2004/DDL-UMamba)
    cd DDL-UMamba/umamba
    pip install -e .
    ```

**Sanity test** — enter the Python command-line interface and run:
```python
import torch
import mamba_ssm
```

## Datasets

We evaluate on two public benchmarks:

| Dataset | Modality | Targets | Split |
| :--- | :--- | :--- | :--- |
| **Synapse** (Dataset017_Synapse) | Abdominal CT | 8 organs: aorta, gallbladder, kidneys (L/R), liver, pancreas, spleen, stomach | 18 train / 12 test (TransUNet split) |
| **ACDC** (Dataset027_ACDC) | Cardiac cine-MRI | 3 structures: RV, MYO, LV | 160 train / 40 test |

Organize the data as follows, following nnU-Net's raw-data convention:

```text
data/
├── nnUNet_raw/
│   ├── Dataset017_Synapse/
│   │   ├── imagesTr/
│   │   │   ├── img0001_0000.nii.gz
│   │   │   ├── img0002_0000.nii.gz
│   │   │   └── ...
│   │   ├── labelsTr/
│   │   │   ├── label0001.nii.gz
│   │   │   ├── label0002.nii.gz
│   │   │   └── ...
│   │   └── dataset.json
│   └── Dataset027_ACDC/
│       ├── imagesTr/
│       │   ├── patient001_frame01_0000.nii.gz
│       │   ├── patient001_frame12_0000.nii.gz
│       │   └── ...
│       ├── labelsTr/
│       │   ├── patient001_frame01.nii.gz
│       │   ├── patient001_frame12.nii.gz
│       │   └── ...
│       └── dataset.json
```

## Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Model training

**Train 3D models**

Train DDL-UMamba_Bot (DDL-CC at the bottleneck only):
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerDDLUMambaBot
```

Train DDL-UMamba_Enc (DDL-CC at every encoder stage, primary model):
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerDDLUMambaEnc
```

**Key hyperparameters used in our experiments** (see the paper for full detail):

| Setting | Synapse | ACDC |
| :--- | :--- | :--- |
| **$d_v$ (expansion factor)** | 4 | 3 |
| **Patch size** | 48×192×192 | 20×256×224 |
| **Batch size** | 2 | 4 |
| **Boundary-spectral loss weight $\gamma$** | 0.1 | 0.1 |
| **$\lambda_{bd} / \lambda_{sp}$** | 0.5 / 0.05 | 0.5 / 0.05 |

## Inference

Predict testing cases with DDL-UMamba_Bot:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerDDLUMambaBot --disable_tta
```

Predict testing cases with DDL-UMamba_Enc:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerDDLUMambaEnc --disable_tta
```

## Evaluation

Evaluation scripts are provided in `evaluation/`, adapted from U-Mamba's original evaluation code:

```bash
python evaluation/Synapse_DSC.py  --gt_path <labelsTs_dir> --seg_path <predictions_dir> --save_path <output.csv>
python evaluation/Synapse_HD95.py --gt_path <labelsTs_dir> --seg_path <predictions_dir> --save_path <output.csv>
python evaluation/ACDC_DSC.py     --gt_path <labelsTs_dir> --seg_path <predictions_dir> --save_path <output.csv>
```

## Results

| Dataset | Model | Mean DSC (%) | HD95 (mm) | Params (M) | FLOPs (G) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Synapse | U-Mamba_Enc (backbone) | 82.83 | 27.17 | — | — |
| Synapse | DDL-UMamba_Enc | **86.93** | **17.56** | 44.61 | 1789.5 |
| ACDC | U-Mamba_Enc (backbone) | 92.22 | — | — | — |
| ACDC | DDL-UMamba_Enc | **92.46** | — | 44.10 | 1167.3 |

DDL-CC adds under 5% parameter overhead to the U-Mamba backbone. The largest per-organ gains on Synapse occur on the gallbladder (+10.13) and pancreas (+14.84) — small, thin-walled, low-contrast structures — with the myocardium showing the largest single-structure gain on ACDC, consistent with the mechanism specifically improving boundary fidelity on thin and low-contrast anatomy.

## Remarks

**Path settings.** The default data directory is preset to `DDL-UMamba/data`. To use a different location, adjust `umamba/nnunetv2/path.py`:
```python
base = '/home/user_name/Documents/DDL-UMamba/data'
nnUNet_raw = join(base, 'nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results')
```

**Mixed precision.** Our models were trained successfully with automatic mixed precision (AMP) enabled. As with the original U-Mamba, AMP has occasionally been reported to produce NaNs in the Mamba module on some setups; if this occurs, disable AMP in your trainer as a fallback.

## Paper

If you find this work useful, please cite:

```bibtex
@article{DDL-UMamba,
    title={DDL-UMamba: Geometric Residual Learning for 3D Medical Image Segmentation},
    author={D. Das, R. Singh, P. Ghosal, A. Pradhan},
    year={2026}
}
```

This work builds on U-Mamba:

```bibtex
@article{U-Mamba,
    title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
    author={Ma, Jun and Li, Feifei and Wang, Bo},
    journal={arXiv preprint arXiv:2401.04722},
    year={2024}
}
```

## Acknowledgements

This repository is built on top of the official U-Mamba codebase; we thank Jun Ma, Feifei Li, and Bo Wang for making it publicly available. We also thank the authors of nnU-Net, Mamba, and Deep Delta Learning for their publicly released code, and the organisers of the Synapse Multi-Atlas Labeling and ACDC challenges for the public datasets.
