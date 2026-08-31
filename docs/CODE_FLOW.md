# Code flow

Module-level detail. For the architecture at a glance — data pipeline, branch
training, fusion selection, and inference — see
[`aigc_architecture.drawio`](../aigc_architecture.drawio) (seven pages: data, branch
training, fusion selection, inference, hardware and frameworks, the degradation
engine, and the notebook-01 training loop).

## Training

```text
configs/recipes/*.yaml
        │
        ▼
Workflows/train.py
        │ builds DataConfig + ExpertConfig + TrainConfig
        ├── Data_Pipeline/datasets.py
        │      manifest row → native crop → clean/degraded tensor pair
        │
        ├── Modeling/dinov3/{backbone,detector,losses}.py
        │      DINOv3 features → optional correction → fake logit → robust loss
        │
        └── Modeling/dinov3/trainer.py
               AdamW + AMP + warmup/cosine + optional EMA/SWA
                         │
                         ▼
               Evaluation/training_validation.py
               clean + degradation grid → robust AUC selection
                         │
             ┌──────────────────────┐
             ▼                      ▼
outputs/runs/<expert>/<time>/   weights/dinov3/<expert>/
config + history               last.pt + best_robust.pt + swa.pt
```

Both views originate from one crop. The degradation is applied after cropping,
so consistency losses cannot exploit a resolution mismatch. Dataset workers
are not persistent because each worker must observe the new curriculum epoch.

`last.pt` is a true resume checkpoint. `best_robust.pt` is a compact inference
checkpoint selected on the degradation grid.

## Evaluation

```text
data/evaluation/*.csv or data/test/manifest.csv
        │
        ▼
Workflows/evaluate.py → Modeling/registry.py → Scorer
        │
        ▼
Evaluation/robustness.py
clean + exact competition cells + held-out cells
        │
        ▼
Evaluation/metrics.py → CSV + PNG + Markdown
```

Community Forensics, trained DINOv3, and diagnostic scorers share this
path. Third-party preprocessing stays inside its model adapter.

## Prediction

```text
predict.py (stable wrapper)
        ▼
Workflows/predict.py
recurse files → Data_Pipeline/image_io.py → registered scorer
        ▼
[{"image_path": ..., "pred": P(AI-generated)}]
```

## Full-pipeline classification

```text
classify.py (user-facing wrapper)
        ▼
Workflows/classify.py
files/folders/stdin/prompt → Data_Pipeline/image_io.py
        │
        │  one detector loaded at a time, scores cached per input set
        ├── community_forensics ──┐
        ├── dinov3 (DINOv3-L) ────┼──► weights/fusion/selected_final_fusion.joblib
        └── stacking (DINOv3-B) ──┘             │
                                                ▼
                             P(fake) ≥ threshold → REAL/FAKE table,
                             optional metrics, CSV, and JSON
```

The fusion artifact's `feature_order` decides which detectors run and in which
order; `BRANCHES` in `Workflows/classify.py` maps each feature name to its
detector. Re-selecting the fusion model changes the CLI without editing it.

## Ownership boundaries

| concern | owner |
|---|---|
| source downloads | `Data_Pipeline/sources/` |
| lifecycle splits and file safety | `Data_Pipeline/` |
| shared device/degradations/scorer contract | `Modeling/common/` |
| one model's architecture/training | its `Modeling/<name>/` folder |
| model-independent metrics and reports | `Evaluation/` |
| CLI orchestration | `Workflows/` |

This boundary is the main defence against duplicated logic: workflows compose
modules, but do not reimplement image loading, device selection, metrics, or
degradations.
