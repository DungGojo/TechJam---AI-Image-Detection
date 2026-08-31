# Model weights

The project expects these runtime artifacts:

```text
weights/
├── dinov3/notebook_dinov3/best_robust.pt
├── stacking/notebook_vitb/best_robust.pt
└── fusion/selected_final_fusion.joblib
```

The small final fusion artifact is tracked by normal Git. The two `.pt`
detector checkpoints are tracked with Git LFS, so teammates should install Git
LFS and run:

```bash
git lfs install
git lfs pull
```

SHA-256 checksums:

```text
55706bcfa7940e4b1f9d27fc33f165034b45d6de4949a3f2e0209179b1eac8df  dinov3/notebook_dinov3/best_robust.pt
023d01c85b0569f5b912ffbe28dd52b13ff9210086e1cc0d0e65cdd7220bd251  stacking/notebook_vitb/best_robust.pt
7390ecbbdaa0018df1e5d280d4b27cb60b0dfdce99d142539ca62b5a5cb2cefe  fusion/selected_final_fusion.joblib
```

The DINOv3 file contains the trained model and its architecture configuration.
The stacking file contains the fitted head and frozen-backbone name. The
backbone and Community Forensics dependency are downloaded into the normal
Hugging Face cache on first use.
