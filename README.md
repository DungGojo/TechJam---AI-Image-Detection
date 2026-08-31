# AI-Generated Image Detection

A complete pipeline for classifying images as authentic (`0`) or AI-generated
(`1`). Three detector probabilities are fused by the final model selected with
cross-validation.

## Start here

1. **Clone with Git LFS and pull weights**:

   ```bash
   git lfs install
   git clone https://github.com/DungGojo/TechJam---AI-Image-Detection.git
   git lfs pull 
   ```

   The small final fusion model is included in Git; the two large `.pt` detector
   checkpoints are pulled via Git LFS. See [`weights/README.md`](weights/README.md).

2. **Set up the virtual environment & install dependencies**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .                     # registers the `classify` CLI command
   ```

3. **Put evaluation images in** (optional):

   ```text
   testing_images/
   ├── real/   # authentic images
   └── fake/   # AI-generated images
   ```

   Local images are intentionally not included in Git.

4. **Classify them from the terminal**:

   ```bash
   classify testing_images --evaluate
   ```

   Or run directly on any image:

   ```bash
   python classify.py photo.jpg
   ```

   Run `classify` with no arguments to be prompted for paths instead. See
   [Classify your own images](#classify-your-own-images) for the full interface.

   Or open `notebooks/Full_Pipeline_Eval_Custom_Folder.ipynb`
   and run all cells.

Notebook predictions, metrics, plots, and review examples are written to
`outputs/full_pipeline/`.

## Architecture

Three detectors score every image independently; a fusion model turns their three
probabilities into one decision.

```text
                    ┌─ Community Forensics ──────── P1 ─┐
                    │  pretrained ViT, no crop TTA      │
  input image ──────┼─ DINOv3-L detector ─────────── P2 ─┼──► SVM (linear)
                    │  8 crops, mean of sigmoids        │    P(fake) ≥ 0.70
                    └─ DINOv3-B stacking ─────────── P3 ─┘         │
                       3 crops + CF logit                          ▼
                                                            REAL / FAKE
```

Each branch is deliberately different: an off-the-shelf pretrained detector, an
expert fine-tuned on the training split, and a frozen backbone whose small head also
consumes the first branch's logit. Their errors are less correlated than three
variants of one model would be, which is what makes fusing them worthwhile.

Full diagrams are in [`aigc_architecture.drawio`](aigc_architecture.drawio) — seven
pages on a white canvas, open with [diagrams.net](https://app.diagrams.net):

| Page | Covers |
|---|---|
| 1. Data and Splits | archives → range-request download → manifest → five splits |
| 2. Branch Training | DINOv3-L expert and the ViT-B stacking head (notebooks 01, 02) |
| 3. Fusion Selection | meta-dataset → ten candidates → 5-fold CV (notebooks 03, 04) |
| 4. Inference | image → three probabilities → fusion → verdict |
| 5. Hardware and Frameworks | what each stage runs on, and which library owns what |
| 6. Degradation Engine | the ten transform families, severity scale, curriculum, and evaluation grid |
| 7. Training Loop | notebook 01 in detail — paired views, the four-term loss, checkpoint selection |

Pages 6 and 7 are the training deep-dive: page 6 covers
`Modeling/common/degradations.py`, page 7 walks
`notebooks/01_Train_DINOv3.ipynb` from manifest to `best_robust.pt`.

`notebook05_full_pipeline.drawio` is kept as the original inference-only diagram.

A visual overview of the same architecture, with the benchmark results, is published
at [Three Branches, One Verdict](https://claude.ai/code/artifact/c453d2e0-bc38-4a57-af81-f1d1b3a9f612).
Its diagram is drawn independently of the `.drawio` file rather than exported from it,
so a change to the architecture needs applying in both places.

The three probability columns and the decision threshold both come from
`weights/fusion/selected_final_fusion.joblib`. Its `feature_order` field decides
which detectors run and in what order, so re-running notebook 04 changes the
pipeline without editing any code.

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_Train_DINOv3.ipynb` | Train or load the DINOv3-L detector |
| `02_Train_ViTB_Stacking.ipynb` | Train or load the frozen ViT-B stacking detector |
| `03_Build_Three_Branch_Meta_Dataset.ipynb` | Build the three-probability fusion dataset |
| `04_Learn_Best_Final_Fusion.ipynb` | Compare fusion models and select the final configuration |
| `05_Full_Pipeline_Image_Classification_(Test Full).ipynb` | Saved full-test execution and results |
| `my-notebook/aigc-detector-benchmark/` | Kaggle kernel: held-out benchmark on GPU |
| `Full_Pipeline_Eval_Custom_Folder.ipynb` | Run the final pipeline on any user-supplied image folder |

Each detector and the final fusion model output `P(AI-generated)`. The selected
fusion threshold is stored with the included fusion artifact.

## Submission structure

```text
Coding/
├── notebooks/              training, fusion, and final-pipeline notebooks
├── my-notebook/            Kaggle kernels (sources + kernel-metadata.json)
├── Modeling/               detector implementations and shared scorer API
├── Data_Pipeline/          image loading, manifests, and data preparation
├── Evaluation/             metrics and visualisation
├── Workflows/              reusable train/evaluate/predict/classify commands
├── configs/                primary recipe and optional ensemble example
├── weights/                checkpoint instructions + small fusion artifact
├── docs/CODE_FLOW.md       module-level flow
├── aigc_architecture.drawio    four-page architecture diagram
├── classify.py             full-pipeline Real/Fake command-line classifier
├── predict.py              stable single-detector inference entry point
├── pyproject.toml          packaging metadata and the `classify` console script
└── requirements.txt
```

Local datasets/images, large checkpoints, pretrained caches, virtual
environments, and generated outputs are intentionally excluded by `.gitignore`.

## Checkpoints

```text
weights/dinov3/notebook_dinov3/best_robust.pt
weights/stacking/notebook_vitb/best_robust.pt
weights/fusion/selected_final_fusion.joblib
```

The two `.pt` files are stored with Git LFS because the DINOv3 checkpoint alone
is approximately 1.2 GB. Install Git LFS before cloning or run `git lfs pull`
afterward. See [`weights/README.md`](weights/README.md) for hashes.

The ViT-B stacking model also loads the released Community Forensics expert and
the pretrained DINOv3 backbone. On a fresh machine, the first run downloads
those assets into `cache/hf`; later runs reuse the cache.

## Classify your own images

`classify.py` runs the same three-detector pipeline as the final notebook and
prints one Real/Fake verdict per image. Point it at any mix of files and
folders; folders are searched recursively.

### Set up once

```bash
cd ~/Documents/AI-Image-Detection
source .venv/bin/activate
pip install -e .
```

The install puts `classify` on your path. Skip it if you prefer, and run
`python classify.py` in place of `classify` everywhere below.

### Run it

Pass any mix of files and folders:

```bash
classify photo.jpg                          # one image
classify ~/Downloads/screenshots            # a folder, searched recursively
classify img1.jpg img2.png ~/Desktop/batch  # several at once
classify testing_images --evaluate          # with ground truth and metrics
```

```text
PREDICT  P(FAKE)  CF      DINO-L  DINO-B  IMAGE
-----------------------------------------------------------
FAKE     0.992    0.637   0.786   0.838   downloads/a.jpg
REAL     0.011    0.028   0.327   0.389   downloads/b.jpg

2 images: 1 FAKE, 1 REAL
```

`P(FAKE)` is the fused verdict; `CF`, `DINO-L`, and `DINO-B` are the three
detector probabilities behind it, so you can see which branch drove a decision.

### Run it with no arguments

Give it no paths and it prompts for them, which is the easiest way to hand it
a few files:

```text
$ classify

Enter image files or folders to classify (drag and drop works).
Press Enter on an empty line when you are done.

path> ~/Downloads/screenshots
path> /Users/you/Desktop/suspect photo.jpg
path>                          <- blank line starts the run
```

Drag a file from Finder straight into the terminal at the `path>` prompt.
Backslash-escaped spaces and quoted paths are both accepted. Paths also arrive
on standard input, so `find . -name '*.png' | classify` works.

### Options

| Option | Effect |
|---|---|
| `--evaluate` | read ground truth from the nearest `real/` or `fake/` parent folder and print accuracy, precision, recall, specificity, F1, MCC, ROC AUC, and the confusion matrix |
| `--csv PATH`, `--json PATH` | write the per-image detector probabilities and verdicts |
| `--threshold FLOAT` | override the decision threshold stored in the fusion artifact |
| `--device`, `--seed`, `--quiet`, `--no-color` | device override, crop seed, and output control |
| `--cache-dir PATH`, `--no-cache` | control the resumable score cache under `outputs/cli_cache/` |

`classify --help` lists them all.

Detector scores are cached per input set, so an interrupted run resumes and a
repeated run is instant. The two DINOv3 branches sample random crops seeded by
an image's position within its batch, so one image's probability can shift
slightly when it is scored alongside a different set of images. The verdict is
stable; the third decimal is not.

### Building a wheel

```bash
pip wheel . --no-deps -w dist/    # -> dist/coding_aigc_detector-0.1.0-py3-none-any.whl
```

The command locates the checkpoints next to the project. After installing that
wheel somewhere else, running it from an unrelated directory needs
`AIGC_HOME=/path/to/AI-Image-Detection`.

## Single-detector prediction

DINOv3:

```bash
python predict.py \
  --input_dir testing_images \
  --output outputs/dinov3_predictions.json \
  --model dinov3 \
  --checkpoint weights/dinov3/notebook_dinov3/best_robust.pt
```

ViT-B stacking:

```bash
python predict.py \
  --input_dir testing_images \
  --output outputs/vitb_predictions.json \
  --model stacking \
  --checkpoint weights/stacking/notebook_vitb/best_robust.pt
```

## Benchmark results

Held-out evaluation on Kaggle, 13,843 images, Tesla T4, 111 minutes. Both classes
are `eval_only` — quarantined from training by the sha256 and phash blocklists.

| | source | n |
|---|---|---|
| REAL | COCO val2017, official release, native resolution | 5,000 |
| FAKE | WildFake `DALLE/Advanced/DALLE3` | 8,843 |

```text
accuracy            0.9724        confusion    pred REAL  pred FAKE
balanced_accuracy   0.9775      actual REAL        4979         21
precision           0.9975      actual FAKE         361       8482
recall              0.9592
specificity         0.9958
mcc                 0.9424
roc_auc             0.9978
```

Solo accuracy per branch: Community Forensics 0.8149, DINOv3-L 0.9582, DINOv3-B
stacking 0.9619. The fused 0.9724 beats every individual branch.

At the stored threshold of 0.70 the system is conservative — it rarely calls a real
photograph fake (specificity 0.996) but misses about 4% of DALL·E 3 images. Mean
P(fake) is 0.019 on reals against 0.962 on fakes, so that gap is a threshold choice
rather than poor separation; ROC AUC 0.9978 says recall can be bought cheaply by
lowering it.

### Why the reals are not WildFake's COCO copy

WildFake ships COCO val2017 downscaled to a uniform 200×200, while its DALL·E 3
images are native 1024×1024. Pairing those makes the classes separable on image
dimensions alone, before any detector looks at a pixel. It is worse than a generic
confound here: `crop_mode` is `random_crop` at 224px, and `Data_Pipeline/datasets.py`
**tiles** any image smaller than the crop — so every real image would be scored as a
repeated mosaic and no fake one would.

Measured on that pairing the system scores 0.995. Against official full-resolution
COCO it scores 0.972. The 0.023 difference is the size artifact, not detection
skill, and the honest number is the lower one.

## Hardware and frameworks

Device selection is automatic and centralised — `Modeling/common/device.py` resolves
CUDA → MPS → CPU, and nothing else in the codebase hardcodes a device.

| Stage | Needs |
|---|---|
| Train DINOv3-L (nb 01) | CUDA GPU. A 300M-parameter backbone plus AMP activations; CPU is impractical |
| Train stacking head (nb 02) — phase 1 | **GPU.** `cache_features.py` runs the frozen ViT-B *and* the CF expert over every image × 3 crops × 3 variants |
| Train stacking head (nb 02) — phase 2 | CPU-feasible. `head.fit()` reads the cached features; only the 768→256 MLP takes gradients |
| Score meta-dataset (nb 03) | GPU: all three detectors over 2,000 images |
| Fit fusion (nb 04) | CPU only. A 2,000×3 table, seconds to run |
| Inference | CUDA, MPS, or CPU |

Inference peak VRAM is **one detector, not three** — each is constructed, used, then
released. Measured throughput: **0.48 s/image on a Tesla T4**, roughly 1.3 s/image on
Apple MPS. Disk at rest is 1.2 GB (DINOv3-L) + 9.8 MB (stacking) + 8 KB (fusion),
plus ~1.5 GB of Hugging Face backbone cache after the first run.

**GPU compatibility is a hard constraint.** Current PyTorch builds ship no kernels
below compute capability 7.0. A Tesla T4 (sm_75) works; a **Tesla P100 (sm_60) fails**
with `CUDA error: no kernel image is available for execution on the device` on every
forward pass. On Kaggle this means `machine_shape` must be `NvidiaTeslaT4`.

| Framework | Owns |
|---|---|
| PyTorch 2.13 · torchvision 0.28 | tensors, autograd, AMP, device placement |
| timm 1.0.29 | both DINOv3 backbones — **`>=1.0.9` is a hard floor**, earlier releases have no DINOv3 definitions |
| transformers 5.16 · huggingface_hub 1.29 | Community Forensics ViT, its processor, model download and cache |
| scikit-learn 1.9 · joblib 1.5 · xgboost 3.2 | fusion model, scaling, cross-validation, every reported metric |
| NumPy 2.4 · pandas 3.0 · Pillow 12.3 · SciPy 1.17 | arrays, manifests, image decode, score caches |
| requests 2.34 · modelscope 1.39 · tqdm 4.70 | ZIP range-request downloads, ModelScope listing, progress |

Versions are those installed in the project `.venv` on Python 3.11.
`requirements.txt` pins lower bounds only, so newer releases are expected to work.

## Recreating the environment

```bash
# 1. Ensure Git LFS is installed and pull model checkpoints
git lfs install
git lfs pull

# 2. Set up Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .              # adds the `classify` command
```

Device selection is automatic: CUDA, then Apple MPS, then CPU. Training the
DINOv3-L branch is intended for a GPU; comparison and inference also work on
Apple silicon and CPU, although they are slower.

## Data and reproducibility

The large local `data/` directory is working data, not submission source code.
Lifecycle manifests record paths, labels, source, generator, hashes, and image
dimensions. Dataset acquisition and manifest rebuilding live in
`Data_Pipeline/`; model code never branches on dataset source.

See `docs/CODE_FLOW.md` for the module flow and `weights/README.md` for the
checkpoint layout.

## License note

The DINOv3 license is retained under `licenses/DINOv3-LICENSE.md`. Released
third-party models remain subject to their original model-card licenses.
