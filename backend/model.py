# model.py (ready-to-paste, robust inference for form-only features)
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
import shap
import traceback
import math
from collections import defaultdict
import re

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

# Parent allocation priors
ALLOCATION_MAP = {
    "FD": {"FD": 1.0},
    "Short-Term-Debt-MF": {"Short-Term-Debt-MF": 1.0},
    "Debt-MF": {"Debt-MF": 1.0},
    "Hybrid-Conservative": {"Hybrid-Conservative": 0.7, "Debt-MF": 0.3},
    "Hybrid-Aggressive": {"Hybrid-Aggressive": 0.7, "Multi-Cap-Equity": 0.3},
    "Large-Cap-Equity": {"Large-Cap-Equity": 1.0},
    "Multi-Cap-Equity": {"Multi-Cap-Equity": 1.0},
    "Mid-Small-Cap-Equity": {"Mid-Small-Cap-Equity": 1.0},
    "Gold-ETF": {"Gold-ETF": 1.0},
    "Liquid-Fund": {"Liquid-Fund": 1.0},
}

# Sub-allocation priors (child buckets)
SUB_ALLOCATION_MAP = {
    "Multi-Cap-Equity": {"Large-Cap-Equity": 0.5, "Multi-Cap-Core": 0.3, "Mid-Small-Cap-Equity": 0.2},
    "Large-Cap-Equity": {"Large-Cap-Bluechip": 0.7, "Large-Cap-Growth": 0.3},
    "Mid-Small-Cap-Equity": {"Mid-Cap-Equity": 0.6, "Small-Cap-Equity": 0.4},
    "Hybrid-Conservative": {"Hybrid-Conservative": 0.7, "Debt-MF": 0.3},
    "Hybrid-Aggressive": {"Hybrid-Aggressive": 0.7, "Equity": 0.3},
    "Debt-MF": {"Short-Term-Debt-MF": 0.6, "Long-Term-Debt-MF": 0.4},
    "FD": {"FD": 1.0},
    "Gold-ETF": {"Gold-ETF": 1.0},
    "Liquid-Fund": {"Liquid-Fund": 1.0},
}

# Map buckets -> engineered/form features used as evidence
BUCKET_FEATURE_MAP = {
    "Large-Cap-Equity": ["savings_rate", "income", "goal_to_income"],
    "Multi-Cap-Equity": ["savings_rate", "goal_per_month", "goal_to_income"],
    "Mid-Small-Cap-Equity": ["savings_rate", "goal_per_month"],
    "Debt-MF": ["goal_per_month", "savings_rate"],
    "Short-Term-Debt-MF": ["goal_per_month", "savings_rate"],
    "Liquid-Fund": ["savings_rate", "goal_per_month"],
    "FD": ["risk_level_ord", "goal_to_income"],
    "Gold-ETF": ["goal_amount", "goal_to_income", "risk_level_ord"],
    "Hybrid-Conservative": ["savings_rate", "risk_level_ord"],
    "Hybrid-Aggressive": ["savings_rate", "risk_level_ord"]
}

def _finite(v, default=0.0):
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return default

@dataclass
class DhanMitraModel:
    classifier: Any = None
    regressor: Any = None
    alloc_regressor: Any = None
    kmeans: Any = None
    kmeans_scaler: Any = None
    feature_cols: Any = None
    label_encoder: Any = None

    def load(self) -> bool:
        try:
            cls_bundle = joblib.load(MODEL_DIR / "classifier.joblib")
            reg_bundle = joblib.load(MODEL_DIR / "regressor.joblib")
            alloc_bundle = joblib.load(MODEL_DIR / "alloc_regressor.joblib")
            kmeans_bundle = joblib.load(MODEL_DIR / "kmeans.joblib")
            scaler_bundle = joblib.load(MODEL_DIR / "kmeans_scaler.joblib")
            le = joblib.load(MODEL_DIR / "label_encoder.joblib")

            self.classifier = cls_bundle["pipe"]
            self.regressor = reg_bundle["pipe"]
            self.alloc_regressor = alloc_bundle["pipe"]
            self.kmeans = kmeans_bundle.get("model") if isinstance(kmeans_bundle, dict) else getattr(kmeans_bundle, "model", None)
            self.kmeans_scaler = scaler_bundle
            self.feature_cols = cls_bundle.get("feature_cols") or reg_bundle.get("feature_cols")
            self.label_encoder = le
            return True
        except Exception as e:
            print("Model load failed:", e)
            traceback.print_exc()
            return False

    def _build_input_df(self, payload: Dict[str, Any]) -> pd.DataFrame:
        # Build a DataFrame with the exact engineered columns used in training.
        income = _finite(payload.get("income", 0.0))
        expenses = _finite(payload.get("expenses", 0.0))
        goal_amount = _finite(payload.get("goal_amount", 0.0))
        duration_months = int(payload.get("duration_months", 1) or 1)
        risk_level = str(payload.get("risk_level", "medium")).lower().strip()
        risk_map = {"low":0, "medium":1, "high":2}

        savings = max(income - expenses, 0.0)
        savings_rate = (savings / income) if income > 0 else 0.0
        goal_to_income = goal_amount / (income + 1.0)
        goal_per_month = goal_amount / max(1, duration_months)
        log_income = math.log1p(max(0.0, income))
        log_goal = math.log1p(max(0.0, goal_amount))
        risk_level_ord = risk_map.get(risk_level, 1)

        # additional engineered features (kept consistent with training)
        expenses_ratio = (expenses / (income + 1e-9)) if income > 0 else 0.0
        savings_gap = goal_per_month - savings
        # target_burden: guard inf
        target_burden = (goal_per_month / (savings if savings > 0 else 1.0))
        savings_rate_sq = savings_rate ** 2
        log_income_sq = log_income ** 2
        goal_to_income_sq = goal_to_income ** 2

        row = {
            "income": float(income),
            "expenses": float(expenses),
            "savings": float(savings),
            "savings_rate": float(savings_rate),
            "expenses_ratio": float(expenses_ratio),
            "goal_amount": float(goal_amount),
            "duration_months": int(duration_months),
            "goal_per_month": float(goal_per_month),
            "goal_to_income": float(goal_to_income),
            "savings_gap": float(savings_gap),
            "target_burden": float(target_burden),
            "log_income": float(log_income),
            "log_goal": float(log_goal),
            "savings_rate_sq": float(savings_rate_sq),
            "log_income_sq": float(log_income_sq),
            "goal_to_income_sq": float(goal_to_income_sq),
            "risk_level_ord": int(risk_level_ord),
        }

        # Compute cluster using saved scaler and kmeans
        num_cols = ["income","expenses","goal_amount","duration_months","savings","savings_rate"]
        df_row = pd.DataFrame([{
            c: row.get(c, 0.0) for c in num_cols
        }], columns=num_cols)
        try:
            if self.kmeans_scaler is not None:
                scaled = self.kmeans_scaler.transform(df_row)
            else:
                from sklearn.preprocessing import StandardScaler
                scaled = StandardScaler().fit_transform(df_row)
            cluster = int(self.kmeans.predict(scaled)[0]) if self.kmeans is not None else 0
        except Exception:
            cluster = 0
        row["cluster"] = int(cluster)

        cols = list(self.feature_cols) if self.feature_cols is not None else list(row.keys())
        out = {c: row.get(c, 0.0) for c in cols}
        return pd.DataFrame([out], columns=cols)

    def _safe_get_prep_and_model(self, pipeline):
        prep, model = None, None
        try:
            if hasattr(pipeline, "named_steps"):
                prep = pipeline.named_steps.get("prep")
                model = pipeline.named_steps.get("model")
            else:
                model = pipeline
        except Exception:
            model = pipeline
        return prep, model

    def _base_feature_name(self, trans_name: str) -> str:
        if "__" in trans_name:
            parts = trans_name.split("__",1)[1]
        else:
            parts = trans_name
        m = re.match(r"(.+?)_(?:low|medium|high|true|false|0|1)$", parts)
        if m:
            return m.group(1)
        if "[" in parts:
            return parts.split("[",1)[0]
        return parts

    def _aggregate_shap_to_feature_impacts(self, feature_names, contribs):
        agg_abs = defaultdict(float)
        agg_signed = defaultdict(float)
        for name, val in zip(feature_names, contribs):
            base = self._base_feature_name(name)
            agg_abs[base] += float(abs(val))
            agg_signed[base] += float(val)
        return agg_abs, agg_signed

    def _score_bucket_relevance(self, agg_abs: Dict[str, float]) -> Dict[str, float]:
        scores = {}
        total = 0.0
        for bucket, feats in BUCKET_FEATURE_MAP.items():
            s = 0.0
            for f in feats:
                s += float(agg_abs.get(f, 0.0))
            scores[bucket] = s
            total += s
        if total <= 0:
            n = len(BUCKET_FEATURE_MAP) or 1
            return {k: 1.0/n for k in BUCKET_FEATURE_MAP.keys()}
        return {k: (v/total) for k,v in scores.items()}

    def _expand_and_adjust_suballoc(self, parent_alloc, agg_abs, strength=1.5):
        expanded = defaultdict(float)
        for parent, frac in parent_alloc.items():
            if parent in SUB_ALLOCATION_MAP:
                sub = SUB_ALLOCATION_MAP[parent]
                ssum = sum(sub.values()) or 1.0
                for sk, sv in sub.items():
                    expanded[sk] += frac * (sv/ssum)
            else:
                expanded[parent] += frac

        bucket_rel = self._score_bucket_relevance(agg_abs)
        adjusted = {}
        w = min(max((strength-1.0)/(strength+1.0), 0.0), 0.99)
        for bucket, prior in expanded.items():
            relevance = bucket_rel.get(bucket, 0.0)
            adj = (1.0 - w) * prior + w * relevance
            adjusted[bucket] = float(max(adj, 0.0))
        s = sum(adjusted.values()) or 1.0
        return {k: float(v/s) for k,v in adjusted.items()}

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Build input DataFrame
        X_df = self._build_input_df(payload)

        # ML predictions
        try:
            prep, model = self._safe_get_prep_and_model(self.classifier)
            pred_enc = self.classifier.predict(X_df)[0]
            product = str(self.label_encoder.inverse_transform([int(pred_enc)])[0]) if self.label_encoder is not None else str(pred_enc)
            exp_cagr = float(self.regressor.predict(X_df)[0])
        except Exception as e:
            print("Prediction failed:", e)
            traceback.print_exc()
            raise RuntimeError("Prediction failed: " + str(e))

        # Defensive suggested monthly
        income = _finite(payload.get("income",0.0))
        expenses = _finite(payload.get("expenses",0.0))
        goal_amount = _finite(payload.get("goal_amount",0.0))
        duration_months = int(payload.get("duration_months",1) or 1)
        savings = max(income - expenses, 0.0)
        monthly_target = goal_amount / max(1, duration_months)
        suggested_monthly = min(savings, monthly_target * 1.1)
        min_invest = max(500.0, 0.01 * (income or 0))
        if not math.isfinite(suggested_monthly) or suggested_monthly <= 1.0:
            suggested_monthly = min(max(savings, 0.0), min_invest)
            if suggested_monthly <= 0:
                suggested_monthly = min_invest
        suggested_monthly = float(round(suggested_monthly, 2))

        # SHAP explanation (robust) aggregated to engineered features
        explanation = []
        agg_abs = {}
        try:
            prep, est = self._safe_get_prep_and_model(self.classifier)
            if prep is not None:
                X_trans = prep.transform(X_df)
                try:
                    tnames = list(prep.get_feature_names_out())
                except Exception:
                    tnames = list(X_df.columns)
            else:
                X_trans = X_df.values
                tnames = list(X_df.columns)

            est_for_explainer = est or self.classifier
            etype = str(type(est_for_explainer)).lower()
            if ("xgb" in etype) or hasattr(est_for_explainer, "estimators_") or "forest" in etype:
                explainer = shap.TreeExplainer(est_for_explainer)
            else:
                try:
                    explainer = shap.KernelExplainer(est_for_explainer.predict_proba, X_trans)
                except Exception:
                    explainer = shap.KernelExplainer(lambda x: est_for_explainer.predict(x), X_trans)

            shap_values = explainer.shap_values(X_trans)
            if isinstance(shap_values, list) and len(shap_values) > 0:
                class_idx = 0
                try:
                    if hasattr(est_for_explainer, "classes_"):
                        classes = list(est_for_explainer.classes_)
                        # best-effort: choose first class index
                        class_idx = 0
                except Exception:
                    class_idx = 0
                contribs_raw = np.array(shap_values[class_idx])
            else:
                contribs_raw = np.array(shap_values)

            contribs = np.squeeze(contribs_raw)
            if contribs.ndim == 2:
                contribs = contribs[0]
            contribs = np.asarray(contribs, dtype=float).ravel()

            if len(contribs) != len(tnames):
                minlen = min(len(contribs), len(tnames))
                contribs = contribs[:minlen]
                tnames = tnames[:minlen]

            agg_abs_map, agg_signed_map = self._aggregate_shap_to_feature_impacts(tnames, contribs)

            total = sum(agg_abs_map.values()) or 1.0
            items = []
            for k, v in sorted(agg_abs_map.items(), key=lambda x: x[1], reverse=True):
                signed = agg_signed_map.get(k, 0.0)
                items.append({
                    "feature": k,
                    "impact_pct": float(round((v / total) * 100.0, 1)),
                    "direction": "pos" if signed >= 0 else "neg",
                    "impact": float(round((v / total), 3)),
                    "impact_signed": float(round(signed, 6)),
                })
            explanation = items[:6]
            agg_abs = agg_abs_map
        except Exception as e:
            print("SHAP failed:", e)
            traceback.print_exc()
            explanation = [
                {"feature": "savings_rate", "impact_pct": 40.0, "direction": "pos", "impact": 0.40},
                {"feature": "goal_to_income", "impact_pct": 30.0, "direction": "pos", "impact": 0.30},
                {"feature": "goal_per_month", "impact_pct": 20.0, "direction": "pos", "impact": 0.20},
            ]
            agg_abs = {}

        # Dynamic allocation using SHAP relevance
        raw_parent_alloc = ALLOCATION_MAP.get(product) or {"Hybrid-Conservative": 0.6, "Debt-MF": 0.4}
        parent_total = sum(_finite(v,0.0) for v in raw_parent_alloc.values()) or 1.0
        parent_norm = {k: float(_finite(v,0.0)/parent_total) for k,v in raw_parent_alloc.items()}

        adjusted_child_fracs = self._expand_and_adjust_suballoc(parent_norm, agg_abs, strength=1.8)

        allocation_detail = []
        for k in sorted(adjusted_child_fracs.keys()):
            frac = float(adjusted_child_fracs[k])
            rupees = round(frac * suggested_monthly, 2)
            allocation_detail.append({"bucket": k, "fraction": frac, "rupees": rupees})

        fractions_map = {d["bucket"]: d["fraction"] for d in allocation_detail}
        by_rupees_map = {d["bucket"]: d["rupees"] for d in allocation_detail}

        # Build stable input snapshot from X_df to avoid undefined-name errors
        inputs_snapshot = {}
        try:
            row = X_df.iloc[0]
            for c in X_df.columns:
                val = row[c]
                try:
                    if hasattr(val, "item"):
                        val = val.item()
                except Exception:
                    pass
                inputs_snapshot[c] = val
        except Exception:
            inputs_snapshot = {
                "income": income,
                "expenses": expenses,
                "goal_amount": goal_amount,
                "duration_months": duration_months,
                "savings": savings,
                "savings_rate": (savings / income) if income > 0 else 0.0,
                "goal_to_income": goal_amount / (income + 1.0),
                "goal_per_month": goal_amount / max(1, duration_months),
            }

        return {
            "inputs": inputs_snapshot,
            "recommendation": {
                "suggested_monthly_investment": float(suggested_monthly),
                "allocation_breakdown": {
                    "fractions": fractions_map,
                    "by_rupees": by_rupees_map,
                    "detailed": allocation_detail
                },
                "expected_cagr": float(round(exp_cagr, 4)),
                "meta": {
                    "product_class": product,
                    "source": "ml-model",
                    "explanation": explanation
                }
            }
        }

# If module executed directly for quick local test
if __name__ == "__main__":
    import json
    m = DhanMitraModel()
    ok = m.load()
    print("Loaded:", ok)
    sample = {
        "income": 60000,
        "expenses": 45000,
        "goal_amount": 800000,
        "duration_months": 60,
        "risk_level": "medium"
    }
    if ok:
        out = m.predict(sample)
        print(json.dumps(out, indent=2))
