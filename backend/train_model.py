# train_model.py (form-only dataset version)
"""
Train script for DhanMitra using only form-provided fields + engineered features.
Expects sample_data/sample_dataset_balanced.csv (already balanced).
Saves artifacts to models/: classifier.joblib, regressor.joblib, alloc_regressor.joblib,
label_encoder.joblib, kmeans.joblib, kmeans_scaler.joblib, feature_cols.json, metrics.json.
"""
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "sample_data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "sample_dataset_balanced.csv"

PRODUCT_BUCKETS = [
    "FD","Short-Term-Debt-MF","Debt-MF","Hybrid-Conservative","Hybrid-Aggressive",
    "Large-Cap-Equity","Multi-Cap-Equity","Mid-Small-Cap-Equity","Gold-ETF","Liquid-Fund"
]

def feature_engineer(df):
    df = df.copy()
    # base derived from form fields
    df["savings"] = (df.get("income",0) - df.get("expenses",0)).astype(float)
    df["savings_rate"] = (df["savings"] / (df["income"].replace(0, np.nan))).fillna(0.0)
    df["goal_to_income"] = df["goal_amount"] / (df["income"].replace(0, np.nan))
    df["goal_to_income"] = df["goal_to_income"].fillna(0.0)
    df["goal_per_month"] = df["goal_amount"] / df["duration_months"].replace(0,1)
    df["log_income"] = np.log1p(df["income"].clip(lower=0).astype(float))
    df["log_goal"] = np.log1p(df["goal_amount"].clip(lower=0).astype(float))
    # map risk ordinal
    risk_map = {"low":0, "medium":1, "high":2}
    df["risk_level_ord"] = df["risk_level"].map(risk_map).fillna(1).astype(int)
    return df

def load_dataset():
    if not CSV_PATH.exists():
        raise SystemExit(f"Dataset not found: {CSV_PATH} — put your balanced CSV at this path.")
    df = pd.read_csv(CSV_PATH)
    # ensure minimal columns exist
    required = ["income","expenses","goal_amount","duration_months","risk_level","recommended_product","expected_return"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing required columns: {missing}")
    df = feature_engineer(df)
    return df

def sanitize_allocs(df):
    alloc_cols = [f"alloc__{p}" for p in PRODUCT_BUCKETS]
    for c in alloc_cols:
        if c not in df.columns:
            df[c] = 0.0
    Y_alloc_df = df[alloc_cols].copy().fillna(0.0).astype(float)
    zero_idx = (Y_alloc_df.sum(axis=1) == 0.0)
    if zero_idx.any():
        # set one-hot to recommended_product where available
        for idx in Y_alloc_df[zero_idx].index:
            prod = df.at[idx, "recommended_product"] if "recommended_product" in df.columns else None
            col = f"alloc__{prod}" if prod and f"alloc__{prod}" in Y_alloc_df.columns else alloc_cols[0]
            Y_alloc_df.loc[idx, :] = 0.0
            Y_alloc_df.at[idx, col] = 1.0
    Y_alloc_df = Y_alloc_df.div(Y_alloc_df.sum(axis=1).replace(0,1), axis=0)
    if Y_alloc_df.isna().any().any():
        raise ValueError("alloc matrix still contains NaN after sanitize")
    return Y_alloc_df

def main():
    df = load_dataset()
    # features we'll use (form fields + engineered)
    feature_cols = [
        "income","expenses","savings","savings_rate",
        "goal_amount","duration_months",
        "goal_to_income","goal_per_month","log_income","log_goal","risk_level_ord"
    ]

    X = df[feature_cols].copy()
    y_cls = df["recommended_product"].astype(str)
    y_reg = df["expected_return"].astype(float)
    Y_alloc_df = sanitize_allocs(df)
    Y_alloc = Y_alloc_df.values

    # label encode product targets
    le = LabelEncoder()
    y_enc = le.fit_transform(y_cls.values)
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    # numeric + categorical
    num_cols = ["income","expenses","savings","savings_rate","goal_amount","duration_months","goal_to_income","goal_per_month","log_income","log_goal","risk_level_ord"]
    cat_cols = []  # since form only has risk_level handled as ordinal, no OHE needed

    # KMeans profiling (fit scaler and kmeans on numeric columns)
    scaler_for_kmeans = StandardScaler().fit(df[num_cols])
    scaled = scaler_for_kmeans.transform(df[num_cols])
    kmeans = KMeans(n_clusters=4, random_state=42).fit(scaled)
    clusters = kmeans.predict(scaled)
    X["cluster"] = clusters

    # preprocessor: numeric scaler only (no OHE to keep feature space small)
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols + ["cluster"]),
    ], remainder="drop")

    # classifiers & regressors
    try:
        from xgboost import XGBClassifier
    except Exception:
        raise RuntimeError("xgboost is required — install with pip install xgboost")

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    exp_reg = RandomForestRegressor(n_estimators=300, random_state=42)
    alloc_base = RandomForestRegressor(n_estimators=300, random_state=42)
    alloc_reg = MultiOutputRegressor(alloc_base, n_jobs=-1)

    cls_pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    exp_pipe = Pipeline([("prep", preprocessor), ("model", exp_reg)])
    alloc_pipe = Pipeline([("prep", preprocessor), ("model", alloc_reg)])

    # train/test split with stratify where safe
    stratify = y_enc if (len(np.unique(y_enc)) > 1 and np.min(np.bincount(y_enc)) >= 2) else None
    X_train, X_test, y_train, y_test, yreg_train, yreg_test, Yalloc_train, Yalloc_test = train_test_split(
        X, y_enc, y_reg, Y_alloc, test_size=0.20, random_state=42, stratify=stratify
    )

    print("Fitting classifier...")
    cls_pipe.fit(X_train, y_train)
    print("Fitting expected-return regressor...")
    exp_pipe.fit(X_train, yreg_train)
    print("Fitting allocation regressor...")
    alloc_pipe.fit(X_train, Yalloc_train)

    # Evaluate
    y_pred = cls_pipe.predict(X_test)
    yreg_pred = exp_pipe.predict(X_test)
    yalloc_pred = alloc_pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    r2 = r2_score(yreg_test, yreg_pred)
    alloc_mse = mean_squared_error(Yalloc_test, yalloc_pred)
    print(f"Metrics -> classifier_acc: {acc:.4f}, expected_return_r2: {r2:.4f}, alloc_mse: {alloc_mse:.6f}")

    try:
        print("Classification report (decoded):")
        decoded_true = le.inverse_transform(y_test)
        decoded_pred = le.inverse_transform(y_pred)
        print(classification_report(decoded_true, decoded_pred))
    except Exception:
        pass

    # Save artifacts
    joblib.dump({"pipe": cls_pipe, "feature_cols": X.columns.tolist()}, MODEL_DIR / "classifier.joblib")
    joblib.dump({"pipe": exp_pipe, "feature_cols": X.columns.tolist()}, MODEL_DIR / "regressor.joblib")
    joblib.dump({"pipe": alloc_pipe, "feature_cols": X.columns.tolist(), "buckets": PRODUCT_BUCKETS}, MODEL_DIR / "alloc_regressor.joblib")
    joblib.dump({"model": kmeans, "num_cols": num_cols}, MODEL_DIR / "kmeans.joblib")
    joblib.dump(scaler_for_kmeans, MODEL_DIR / "kmeans_scaler.joblib")
    (MODEL_DIR / "feature_cols.json").write_text(json.dumps(X.columns.tolist(), indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"classifier_accuracy": float(acc), "regressor_r2": float(r2), "alloc_mse": float(alloc_mse)}, indent=2))

    print("Saved models and metrics to models/")

if __name__ == "__main__":
    main()
