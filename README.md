# U-DeepBN
Code for paper: "Interpretable AI for Animal Health: Bayesian Networks with Uncertainty-Aware Decision Making"

This repository contains a three-step pipeline for image feature extraction, feature uncertainty estimation, and Bayesian Network model evaluation.

Run the scripts in this order:

1. `feature_extraction.py`
2. `compute_uncertainty.py`
3. `main.py`

## Files

| File | Purpose |
|---|---|
| `feature_extraction.py` | Extracts image features from train/test image folders and saves tabular CSV files. Supports deep ResNet features, classic image features, or hybrid features. |
| `compute_uncertainty.py` | Computes per-feature uncertainty across multiple CSV runs, including aleatoric uncertainty, epistemic uncertainty, total uncertainty, and subjective-logic uncertainty. |
| `main.py` | Trains and evaluates Bayesian Network classifiers using the extracted features and uncertainty-aware TAN modeling. |
| `requirements.txt` | Python package dependencies. |

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Recommended packages:

```txt
numpy
pandas
scipy
scikit-learn
torch
torchvision
Pillow
opencv-python
networkx
```

## Expected data structure

The image dataset should be organized by class folders. For example:

```text
project/
├── train_images/
│   ├── class_1/
│   │   ├── image001.jpg
│   │   └── image002.jpg
│   └── class_2/
│       ├── image003.jpg
│       └── image004.jpg
├── test_images/
│   ├── class_1/
│   └── class_2/
├── feature_extraction.py
├── compute_uncertainty.py
├── main.py
└── requirements.txt
```

The class label is taken from the folder name.

## Step 1: Feature extraction

Run `feature_extraction.py` first to convert images into CSV feature tables.

Example:

```bash
python feature_extraction.py \
  --data train_images \
  --test-dir test_images \
  --out-dir runs_new \
  --features hybrid \
  --deep-backbone resnet18 \
  --deep-dim 128 \
  --topk 0 \
  --aug 0 \
  --seed 42
```

This script saves files like:

```text
runs_new/train_tabular_42.csv
runs_new/test_tabular_42.csv
runs_new/mi_train.csv
```

Useful options:

| Argument | Description |
|---|---|
| `--features` | Feature type: `deep`, `classic`, or `hybrid`. |
| `--deep-backbone` | Deep model backbone: `resnet18` or `resnet50`. |
| `--deep-dim` | Number of PCA dimensions for deep features. |
| `--topk` | Keep top-k features by mutual information. Use `0` to keep all features. |
| `--aug` | Number of augmented copies per training image. |
| `--aug-map` | JSON dictionary for class-specific augmentation counts. |
| `--seed` | Random seed used in output filenames and augmentation. |

To generate multiple runs for uncertainty estimation, run this script multiple times with different seeds:

```bash
python feature_extraction.py --data train_images --test-dir test_images --out-dir runs_new --features hybrid --seed 1
python feature_extraction.py --data train_images --test-dir test_images --out-dir runs_new --features hybrid --seed 2
python feature_extraction.py --data train_images --test-dir test_images --out-dir runs_new --features hybrid --seed 3
```

## Step 2: Compute feature uncertainty

After feature extraction, run `compute_uncertainty.py` on the generated training CSV files.

Example:

```bash
python compute_uncertainty.py \
  --input_dir runs_new \
  --pattern "train*.csv" \
  --exclude_cols label,image_path \
  --bins 10 \
  --per_class_label label \
  --output feature_uncertainty_summary.csv \
  --output_dir subjective_logic_profiles \
  --per_class_output feature_uncertainty_per_class_dog_test.csv
```

This produces:

```text
feature_uncertainty_summary.csv
feature_uncertainty_per_class_dog_test.csv
subjective_logic_profiles/
```

The uncertainty outputs include:

| Column | Meaning |
|---|---|
| `aleatoric` | Average within-run feature variance. |
| `epistemic` | Variance of feature means across runs. |
| `total` | Sum of aleatoric and epistemic uncertainty. |
| `subjective_u` | Subjective-logic uncertainty value. |
| `Hhat` | Normalized entropy of the average feature histogram. |
| `JS` | Jensen-Shannon disagreement across runs. |
| `N_eff` | Effective sample count. |

Note: `main.py` currently expects the uncertainty file name `feature_uncertainty_per_class_dog_test.csv` in the working directory for uncertainty-aware TAN. If you use a different filename, update the default `uncertainty_csv_path` inside `train_tan_uncertainty()` in `main.py`.

## Step 3: Train and evaluate the model

Run `main.py` after creating the feature CSV files and uncertainty CSV file.

Example:

```bash
python main.py \
  --data train_images \
  --test-dir test_images \
  --out-dir results \
  --features hybrid \
  --structure tan \
  --bins 8 \
  --alpha 1.0
```

Important: `main.py` currently reads tabular feature files from the hardcoded folder:

```python
folder = "runs_new"
```

So make sure your feature extraction output directory is named `runs_new`, or edit this line in `main.py`.

Supported model structures:

| Structure | Description |
|---|---|
| `naive` | Gaussian Naive Bayes baseline. |
| `chowliu` | Chow-Liu Bayesian Network. |
| `tan` | Tree-Augmented Naive Bayes, using uncertainty-aware training. |
| `k2` | K2 Bayesian Network structure learning. |

The script prints classification metrics, including per-class accuracy, overall accuracy, confusion matrix information, and classification report.

## Full pipeline example

```bash
# 1. Extract features
python feature_extraction.py \
  --data train_images \
  --test-dir test_images \
  --out-dir runs_new \
  --features hybrid \
  --seed 42

# 2. Compute uncertainty
python compute_uncertainty.py \
  --input_dir runs_new \
  --pattern "train*.csv" \
  --per_class_label label \
  --per_class_output feature_uncertainty_per_class_dog_test.csv

# 3. Train/evaluate model
python main.py \
  --data train_images \
  --test-dir test_images \
  --out-dir results \
  --features hybrid \
  --structure tan
```

## Notes

- Use the same train/test folder structure for all runs.
- For uncertainty estimation, multiple `train*.csv` files are recommended.
- If using `--features deep` or `--features hybrid`, the first run may download pretrained ResNet weights through `torchvision`.
- If CUDA is available, feature extraction will automatically use GPU; otherwise, it uses CPU.
- Keep `label` and `image_path` columns in the CSV files because later scripts use them.

