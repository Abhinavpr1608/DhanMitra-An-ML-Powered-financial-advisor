# explain_instance.py
"""
Safe test harness to construct an input using saved "feature_cols" from classifier.joblib,
call the pipeline, print prediction + probabilities, and compute SHAP contributions safely.
"""

import joblib
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CLS_PATH = MODEL_DIR / "classifier.joblib"

# Example payload: edit values to test different inputs
payload = {
    "income": 80000,
    "expenses": 60000,
    "goal_amount": 220000,
    "duration_months": 36,
    "risk_level": "medium",
    "employment": "salaried",
    "financial_health": "good",
    "age": 35,
    "dependents": 1
}


def build_X_df_from_payload(cls_bundle, payload):
    # Attempt to use saved feature order; fallback to a sensible default
    saved_cols = cls_bundle.get("feature_cols")
    if not saved_cols:
        saved_cols = [
            "income", "expenses", "savings", "savings_rate", "goal_amount",
            "duration_months", "age", "dependents", "employment", "risk_level",
            "financial_health", "equity", "debt", "hybrid", "gold", "cash", "cluster"
        ]

    # Derived features
    income = float(payload.get("income", 0.0))
    expenses = float(payload.get("expenses", 0.0))
    savings = float(max(income - expenses, 0.0))
    savings_rate = float(savings / income) if income > 0 else 0.0

    base_row = {
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "savings_rate": savings_rate,
        "goal_amount": float(payload.get("goal_amount", 0.0)),
        "duration_months": int(payload.get("duration_months", 12)),
        "age": int(payload.get("age", 0)),
        "dependents": int(payload.get("dependents", 0)),
        # categorical defaults (strings) - important for OneHotEncoder
        "employment": str(payload.get("employment", "salaried")),
        "risk_level": str(payload.get("risk_level", "medium")),
        "financial_health": str(payload.get("financial_health", "fair")),
        # placeholder investment buckets if present in features
        "equity": float(payload.get("equity", 0.0)),
        "debt": float(payload.get("debt", 0.0)),
        "hybrid": float(payload.get("hybrid", 0.0)),
        "gold": float(payload.get("gold", 0.0)),
        "cash": float(payload.get("cash", 0.0)),
        # cluster placeholder (if your pipeline / model computes cluster separately)
        "cluster": int(payload.get("cluster", 0)),
    }

    # Ensure every saved column exists; fill missing with safe defaults
    final = {c: base_row.get(c, 0) for c in saved_cols}

    # Cast categorical columns to strings to avoid OneHotEncoder dtype mismatch
    cat_defaults = {"employment", "risk_level", "financial_health"}
    for c in final:
        if c in cat_defaults:
            final[c] = str(final[c])
        else:
            # numeric cast where possible
            try:
                final[c] = float(final[c])
            except Exception:
                try:
                    final[c] = int(final[c])
                except Exception:
                    final[c] = 0.0

    df = pd.DataFrame([final], columns=saved_cols)
    return df


def scalar_from_shap_element(x):
    """Collapse a possibly-array SHAP element into a scalar.
    We preserve direction using signed sum; for raw importance use abs(sum)."""
    a = np.asarray(x)
    if a.size == 0:
        return 0.0
    if a.size == 1:
        return float(a.item())
    # signed sum preserves direction. Use np.abs(a).sum() if you want magnitude-only collapse.
    return float(a.sum())


def main():
    try:
        cls_bundle = joblib.load(CLS_PATH)
    except Exception as e:
        print("Failed to load classifier.joblib:", e)
        return

    pipe = cls_bundle["pipe"]
    print("Pipeline steps:", pipe.named_steps.keys())

    X_df = build_X_df_from_payload(cls_bundle, payload)
    print("Constructed input DF:\n", X_df.to_dict(orient="records")[0])

    # prediction
    try:
        pred = pipe.predict(X_df)[0]
        print("Predicted class:", pred)
    except Exception as e:
        print("Prediction failed (transform error):", e)
        traceback.print_exc()
        return

    # probabilities if available
    try:
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_df)[0]
            model = pipe.named_steps.get("model") or list(pipe.named_steps.values())[-1]
            classes = list(model.classes_) if hasattr(model, "classes_") else None
            if classes:
                print("Class probabilities:")
                for c, p in zip(classes, probs):
                    print(f"  {c}: {p:.4f}")
    except Exception as e:
        print("predict_proba failed:", e)
        traceback.print_exc()

    # attempt SHAP
    try:
        prep = pipe.named_steps.get("prep") or pipe.named_steps.get("preprocessor")
        model = pipe.named_steps.get("model") or list(pipe.named_steps.values())[-1]

        if prep is not None:
            X_trans = prep.transform(X_df)
            try:
                fnames = list(prep.get_feature_names_out(X_df.columns))
            except Exception:
                # fallback: use transformer pieces or original columns
                fnames = []
                if hasattr(prep, "transformers_"):
                    for _, trans, cols in prep.transformers_:
                        try:
                            out = trans.get_feature_names_out(cols)
                            fnames.extend(list(out))
                        except Exception:
                            if isinstance(cols, (list, tuple, np.ndarray)):
                                fnames.extend([str(c) for c in cols])
                            else:
                                fnames.append(str(cols))
                if len(fnames) == 0:
                    fnames = X_df.columns.tolist()
        else:
            X_trans = X_df.values
            fnames = X_df.columns.tolist()

        # Use TreeExplainer for tree models
        try:
            import shap
            expl = shap.TreeExplainer(model)
        except Exception:
            expl = None

        if expl is None:
            print("No appropriate SHAP explainer available for this model.")
            return

        shap_vals = expl.shap_values(X_trans)
        print("SHAP computed (raw)")

        # convert to contrib 1D vector and collapse array-valued entries to scalars
        if isinstance(shap_vals, list) and len(shap_vals) > 0:
            # classification: choose predicted class index
            classes = list(model.classes_) if hasattr(model, "classes_") else None
            class_idx = classes.index(pred) if classes and pred in classes else 0
            contribs_raw = np.array(shap_vals[class_idx])
        else:
            contribs_raw = np.array(shap_vals)

        contribs = np.squeeze(contribs_raw)
        if contribs.ndim == 2:
            contribs = contribs[0]

        # ensure same length as feature names
        contribs_list = [scalar_from_shap_element(x) for x in np.asarray(contribs, dtype=object).ravel()]
        minlen = min(len(contribs_list), len(fnames))
        flattened = [(fnames[i], contribs_list[i]) for i in range(minlen)]

        # sort by absolute contribution
        pairs = sorted(flattened, key=lambda t: abs(t[1]), reverse=True)[:12]
        print("Top contributions (feature: shap_value, abs):")
        for f, v in pairs:
            print(f"  {f}: {v:.6f}    (abs={abs(v):.6f})")

    except Exception as e:
        print("SHAP failed:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
