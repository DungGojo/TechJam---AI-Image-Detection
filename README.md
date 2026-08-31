# AI-Generated Image Detection

A production-grade, highly robust machine learning pipeline for classifying
images as **Real (`0`)** or **AIGC-Generated (`1`)**.

The system leverages a tri-branch architecture combining off-the-shelf forensic
representations, fine-tuned large Vision Transformers with multi-crop Test-Time
Augmentation (TTA), and frozen multi-scale feature stacking, fused via a
cross-validated Linear SVM model.

---

## Architecture Overview

![Architecture Overview](assets/inference_simple.png)

### Tri-Expert Design

1. **Expert 1 (CommunityForensics-DeepfakeDet-ViT)**: An independent off-the-shelf
   forensic baseline extracting uncalibrated artifact signals.
2. **Expert 2 (Frozen DINOv3-Large + Detector Head)**: A frozen 300M-parameter
   DINOv3-ViT-L backbone with Detector Head trained using a 4-term robust
   objective (Clean Focal + Degraded Focal + Prediction Bernoulli KL Divergence
   - Feature MSE) with Attention Pooling and 8-crop test-time aggregation.
3. **Expert 3 (Frozen DINOv3-Base + Stacking Head)**: A frozen 85M-parameter
   DINOv3-ViT-B backbone with a lightweight multi-layer perceptron trained
   across degradation augmentations, conditioned on Expert 1 forensic logits.
4. **Hierarchical Fusion**: A calibrated Linear Support Vector Machine fusing
   the three probability branches to produce the final $P(\text{AI-Generated})$
   verdict.

---

## Setup and Installation

### 1. Prerequisites & Git LFS (Large File Storage)

The repository uses Git LFS to track model checkpoints (`checkpoints/**/*.pt`).
Ensure Git LFS is installed before pulling weights:

- Ubuntu / Debian:
  ```bash
  sudo apt-get update && sudo apt-get install -y git-lfs
  ```
- macOS (Homebrew):
  ```bash
  brew install git-lfs
  ```
- Clone Repository and Pull Large Checkpoints:
  ```bash
  git lfs install
  git clone https://github.com/DungGojo/TechJam---AI-Image-Detection.git
  cd TechJam---AI-Image-Detection
  git lfs pull
  ```

---

### 2. Environment Setup with `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package
manager.

- Install `uv`:
  - Linux / macOS (curl):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  - macOS (Homebrew):
    ```bash
    brew install uv
    ```

- Create Environment & Install Dependencies:
  ```bash
  uv venv .venv --python 3.11
  source .venv/bin/activate
  ```
- Install Dependencies:

  ```
  uv sync
  ```

  Or with `--dev` for downloading datasets, training and testing

  ```
  uv sync --dev

  ```

---

### 3. Fallback Environment Setup with `pip`

If you prefer standard `pip`:

- Create and activate virtual environment

  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  ```

- Install dependencies

  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

---

## Running Inference & Predictions

### A. Run Inference

Score an entire directory of images and export confidence scores to JSON,
change `path/to/data` to the image directory path:

```bash
python scripts/run_inference.py \
  --input path/to/data/ \
  --output outputs/predictions.json
```

**Output JSON Structure**: `pred` score indicating the likelihood that the
image is AIGC-generated range from 0 to 1, with 1 being most confidence that
the image is AIGC-generated.

```json
[
  {
    "image_path": "path/to/data/sample_01.jpg",
    "pred": 0.009993
  },
  {
    "image_path": "path/to/data/sample_02.png",
    "pred": 0.99999
  }
]
```

---

### B. Ground-Truth evaluation

Evaluate ground-truth labels inferred from parent folder names (`real/` vs
`fake/`) and report full metrics:

```bash
python scripts/run_inference.py \
  --input path/to/data/ \
  --evaluate
```

---

### C. Export to CSV

Export per-image prediction probabilities and individual detector branch scores
to CSV:

```bash
python scripts/run_inference.py \
  --input path/to/data/ \
  --output outputs/predictions.csv
```

---

## Steps to Reproduce Results

### Step 1: Data Acquisition & Manifest Creation

Download the training, validation, and evaluation datasets. Manifests record
paths, ground-truth labels, SHA-256 hashes, and perceptual hashes (pHash) to
strictly prevent data leakage.

```bash
# 1. Download SID_Set training data
python -m data_pipeline.sources.download_sid_set

# 2. Download WildFake generator splits via HTTP range requests
python -m data_pipeline.sources.download_wildfake

# 3. Download official COCO val2017 & DALL-E evaluation benchmarks
python -m data_pipeline.sources.download_eval_only
```

---

### Step 2: Train Expert 2 (Frozen DINOv3-Large + Detector Head)

Frozen DINOv3-Large with detector trained using Attention Pooling, 6-family
degradation curriculum, and the 4-term robust objective:

- **Command Line**:
  ```bash
  python -m workflows.train --recipe configs/recipes/ours.yaml
  ```
- **Output Checkpoint**: `checkpoints/dinov3/notebook_dinov3/best_robust.pt`

---

### Step 3: Train Expert 3 (Frozen DINOv3-Base + Feature Caching & Stacking Head)

Cache frozen multi-crop DINOv3-Base embeddings to memory-mapped files and train
the routing Stacking Head:

- **Command Line**:
  ```bash
  # 1. Extract and cache features across clean and degraded variants
  python -m workflows.cache_plan configs/cache_plan.vitb.yaml

  # 2. Fit the Stacking MLP head
  python -m workflows.fit_stacker \
    --checkpoint checkpoints/stacking/notebook_vitb/best_robust.pt
  ```
- **Output Checkpoint**: `checkpoints/stacking/notebook_vitb/best_robust.pt`

---

### Step 4: Meta-Dataset Assembly & Fusion Selection

Assemble the 3-detector probability matrix and select the optimal fusion algorithm:

1. **Generate Meta-Dataset (CLI)**:
   Score validation images across all three detector branches and export the tabular meta-dataset:

   ```bash
   python scripts/run_inference.py \
     --input data/evaluation/images/validation \
     --csv outputs/meta_dataset.csv
   ```

2. **Train & Select Fusion Model**:
   Compare classifier families (Linear SVM, Logistic Regression, XGBoost,
   Decision Trees) across 5-fold cross-validation in
   [`notebooks/04_Learn_Best_Final_Fusion.ipynb`](notebooks/04_Learn_Best_Final_Fusion.ipynb).
   - **Output Artifact**: `checkpoints/fusion/selected_final_fusion.joblib`
     (Calibrated Linear SVM, decision threshold $\tau = 0.70$).

---

### Step 5: Benchmark Evaluation & Analysis

Evaluate the final fused pipeline across clean and degraded benchmarks:

1. **Benchmark Evaluation**:
   Run full tri-expert evaluation on the 13,843 quarantined benchmark images:

   ```bash
   python scripts/run_inference.py \
     --input data/evaluation/images/benchmark \
     --evaluate
   ```

2. **Robustness Degradation Grid Analysis**:
   Generate degradation performance curves across JPEG, Blur, Resize, Noise,
   Jitter, and Crop transformations in
   [`notebooks/07_Demo_Result.ipynb`](notebooks/07_Demo_Result.ipynb).

3. **Performance Analysis**:
   False Positive and False Negative analysis in
   [`notebooks/08_Performance_Analysis.ipynb`](notebooks/08_Performance_Analysis.ipynb).

---

## Benchmark Performance

Evaluated on 13,843 quarantined benchmark images (5,000 Real COCO val2017 + 8,843 Fake DALL·E 3):

| Metric                                     | Score (Clean Images) | Score (Transformed Images) |
| ------------------------------------------ | -------------------- | -------------------------- |
| **ROC AUC**                                | **99.96%**           | **99.62%**                 |
| **Accuracy**                               | **97.32%**           | **94.63%**                 |
| **F1**                                     | **97.86%**           | **95.49%**                 |
| **Precision**                              | **99.90%**           | **99.12%**                 |
| **Recall / Sensitivity**                   | **95.90%**           | **92.46%**                 |
| **Specificity**                            | **99.83%**           | **98.47%**                 |
| **Matthews Correlation Coefficient (MCC)** | **94.43%**           | **89.54%**                 |

---

## Directory Structure

```text
.
├── data_pipeline/          # Image loading, dataset downloads, pHash/SHA256 manifests
├── evaluation/             # Metric computations, degradation grids, visualization
├── modeling/               # PyTorch architectures (DINOv3, Stacking, Fusion, CF)
├── workflows/              # CLI workflows (train, evaluate, classify, predict, cache)
├── configs/                # Global configurations (default.yaml, recipes/ours.yaml)
├── notebooks/              # Step-by-step training, fusion, and evaluation notebooks
├── scripts/
│   └── run_inference.py    # Inference and evaluation script
├── checkpoints/            # Checkpoints (Expert 2, Expert 3, Linear SVM parameters)
├── pyproject.toml          # Packaging metadata & dependency groups
├── requirements.txt        # Dependencies requirements for `pip`
├── LICENSE.md              # Project and model licensing
└── README.md
```

---

## License

- DINOv3 model definitions and weights are governed by the license under `LICENSE.md`.
- Pretrained baseline models belong to their respective creators under original model-card licenses.
