# map_feature_importances.py
import joblib
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
cls_bundle = joblib.load(MODEL_DIR / "classifier.joblib")
pipe = cls_bundle["pipe"]
prep = pipe.named_steps.get("prep") or pipe.named_steps.get("preprocessor")
model = pipe.named_steps.get("model") or list(pipe.named_steps.values())[-1]

print("Estimator:", type(model))
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
else:
    raise SystemExit("Model has no feature_importances_")

# get transformed feature names
try:
    feat_names = list(prep.get_feature_names_out())
except Exception:
    # fallback: build from transformers_
    feat_names = []
    for name, transformer, cols in prep.transformers_:
        if transformer == "drop":
            continue
        try:
            out = transformer.get_feature_names_out(cols)
            feat_names.extend(list(out))
        except Exception:
            if isinstance(cols, (list, tuple)):
                feat_names.extend([str(c) for c in cols])
            else:
                feat_names.append(str(cols))

# Trim or extend feat_names to match importances length
if len(feat_names) < len(importances):
    # pad with indexes
    feat_names += [f"t_{i}" for i in range(len(feat_names), len(importances))]
elif len(feat_names) > len(importances):
    feat_names = feat_names[:len(importances)]

# print top n
n = 30
idx = np.argsort(importances)[::-1][:n]
print(f"Top {n} transformed features by importance:")
for i in idx:
    print(f"{i:3d}: {feat_names[i]:40s} → {importances[i]:.4f}")
