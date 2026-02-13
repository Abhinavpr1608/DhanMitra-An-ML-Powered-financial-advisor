# inspect_classifier.py (fixed: builds engineered features & cluster before using feature_cols)
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "sample_data"

# helpers: same feature engineering as training
def feature_engineer(df):
    df = df.copy()
    df['goal_to_income'] = df['goal_amount'] / (df['income'] + 1.0)
    df['goal_per_month'] = df['goal_amount'] / df['duration_months'].replace(0, 1)
    df['debt_to_equity'] = df['debt'] / (df['equity'] + 1.0)
    df['log_income'] = np.log1p(df['income'])
    df['log_goal'] = np.log1p(df['goal_amount'])
    risk_map = {"low": 0, "medium": 1, "high": 2}
    fin_map = {"poor": 0, "average": 1, "good": 2, "excellent": 3}
    df['risk_level_ord'] = df['risk_level'].map(risk_map).fillna(1).astype(int)
    df['financial_health_ord'] = df['financial_health'].map(fin_map).fillna(1).astype(int)
    return df

def compute_cluster_rowwise(df, num_cols, kmeans, scaler):
    # Build numeric matrix and transform using scaler; then predict cluster
    Xnum = df[num_cols].copy().astype(float).fillna(0.0)
    if scaler is not None:
        try:
            Xs = scaler.transform(Xnum)
        except Exception:
            # fallback: fit a local scaler (not ideal)
            from sklearn.preprocessing import StandardScaler
            s = StandardScaler().fit(Xnum)
            Xs = s.transform(Xnum)
    else:
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler().fit(Xnum)
        Xs = s.transform(Xnum)
    if kmeans is not None:
        try:
            clusters = kmeans.predict(Xs)
        except Exception:
            # fallback: use zeros
            clusters = np.zeros(Xnum.shape[0], dtype=int)
    else:
        clusters = np.zeros(Xnum.shape[0], dtype=int)
    return clusters

# Load classifier and label encoder
cls_bundle = joblib.load(MODEL_DIR / "classifier.joblib")
pipe = cls_bundle.get("pipe") if isinstance(cls_bundle, dict) else cls_bundle
feature_cols = cls_bundle.get("feature_cols") if isinstance(cls_bundle, dict) else getattr(cls_bundle, "feature_cols", None)

try:
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    class_labels = list(le.classes_)
except Exception:
    le = None
    est = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
    class_labels = [str(c) for c in getattr(est, "classes_", [])]

print("Learned product classes:", class_labels)
print("Expected feature_cols length:", len(feature_cols) if feature_cols is not None else "unknown")

# Load dataset (balanced CSV you renamed)
csv_path = DATA_DIR / "sample_dataset_balanced.csv"
if not csv_path.exists():
    csv_path = DATA_DIR / "sample_dataset_balanced_3256.csv" if (DATA_DIR / "sample_dataset_balanced_3256.csv").exists() else csv_path

if not csv_path.exists():
    raise SystemExit(f"Cannot find dataset at {csv_path}. Place your balanced CSV into sample_data/ and name it sample_dataset_balanced.csv")

df_raw = pd.read_csv(csv_path)
print("Loaded CSV rows:", len(df_raw))

# Ensure required base columns exist; if missing, raise friendly error
required_base = ["income","expenses","savings","savings_rate","goal_amount","duration_months","age","dependents","employment","risk_level","financial_health","equity","debt","hybrid","gold","cash"]
missing_base = [c for c in required_base if c not in df_raw.columns]
if missing_base:
    raise SystemExit(f"Dataset missing required columns: {missing_base}")

# Apply feature engineering
df = feature_engineer(df_raw)

# Compute cluster using saved kmeans and scaler if available
kmeans = None
scaler_for_kmeans = None
try:
    km_bundle = joblib.load(MODEL_DIR / "kmeans.joblib")
    if isinstance(km_bundle, dict):
        kmeans = km_bundle.get("model")
    else:
        kmeans = getattr(km_bundle, "model", None)
except Exception:
    kmeans = None

try:
    scaler_for_kmeans = joblib.load(MODEL_DIR / "kmeans_scaler.joblib")
except Exception:
    scaler_for_kmeans = None

# determine numeric columns used for kmeans (same as training)
num_cols = ["income", "expenses", "goal_amount", "duration_months", "savings", "savings_rate"]
# ensure they exist
num_cols = [c for c in num_cols if c in df.columns]
df['cluster'] = compute_cluster_rowwise(df, num_cols, kmeans, scaler_for_kmeans)

# Now we should have all engineered columns; verify feature_cols exist in df
missing_feats = [c for c in feature_cols if c not in df.columns]
if missing_feats:
    print("After engineering, still missing feature columns:", missing_feats)
    # as a fallback, create missing numeric features as zeros and missing categorical as 'unknown'
    for c in missing_feats:
        if c in ["employment","risk_level","financial_health"]:
            df[c] = df.get(c, "unknown").astype(str)
        else:
            df[c] = df.get(c, 0.0).astype(float)
    print("Filled missing feature columns with defaults.")

# Sample 1000 rows for prediction distribution
sample = df.sample(min(1000, len(df)), random_state=42)
X = sample[feature_cols]

# Predict
preds = pipe.predict(X)
decoded = le.inverse_transform(preds) if le is not None else preds

print("\nPrediction distribution on sample:")
print(pd.Series(decoded).value_counts())

# Show top importances if available
est = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
if hasattr(est, "feature_importances_"):
    importances = est.feature_importances_
    # If importances length equals transformed feature space, we try to map to numeric feature names
    try:
        prep = pipe.named_steps.get("prep")
        try:
            tnames = list(prep.get_feature_names_out())
        except Exception:
            tnames = None
    except Exception:
        tnames = None

    top_idx = np.argsort(importances)[::-1][:20]
    print("\nTop feature importances (index:name:importance):")
    for i in top_idx:
        name = tnames[i] if (tnames is not None and i < len(tnames)) else (feature_cols[i] if i < len(feature_cols) else str(i))
        print(f"  {i}: {name}: {importances[i]:.4f}")
else:
    print("\nModel has no feature_importances_ attribute.")
