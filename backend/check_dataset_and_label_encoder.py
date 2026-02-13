# check_dataset_and_label_encoder.py
import pandas as pd, joblib, numpy as np
from pathlib import Path
MODEL_DIR = Path("models")
DATA_DIR = Path("sample_data")

# 1) dataset counts
for f in ["sample_dataset_balanced.csv","sample_dataset.csv","sample_dataset_10000.csv"]:
    p = DATA_DIR / f
    if p.exists():
        df = pd.read_csv(p)
        if "recommended_product" in df.columns:
            print(f"\n{p.name} rows:{len(df)} unique products:{df['recommended_product'].nunique()}")
            print(df['recommended_product'].value_counts())
        else:
            print(f"\n{p.name} — no recommended_product column")
    else:
        print(f"\n{p.name} — not found")

# 2) show saved label encoder classes (decoded)
le_path = MODEL_DIR / "label_encoder.joblib"
if le_path.exists():
    le = joblib.load(le_path)
    print("\nLabelEncoder.classes_ (decoded):", list(le.classes_))
else:
    print("\nlabel_encoder.joblib not found in models/")

# 3) sanity: show classifier.classes_ (raw) and map to decoded if possible
cls_bundle = joblib.load(MODEL_DIR / "classifier.joblib")
pipe = cls_bundle.get("pipe") if isinstance(cls_bundle, dict) else cls_bundle
est = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
print("\nEstimator.classes_ (raw):", getattr(est, "classes_", None))
try:
    if le_path.exists():
        decoded = le.inverse_transform(np.arange(len(getattr(est,"classes_",[]))))
        print("Decoded labels for estimator.classes_ indices:", list(decoded))
except Exception as e:
    print("Could not decode estimator.classes_ indices:", e)
