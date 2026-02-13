# fix_balance_add_missing.py
"""
Reads sample_data/sample_dataset_balanced.csv, checks which of the 10 desired product
labels are missing, generates synthetic rows for missing labels to equalize counts
to the dataset max (or to a user-specified target), and writes sample_data/sample_dataset_balanced_fixed.csv.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")
DATA_DIR = ROOT / "sample_data"
IN_PATH = DATA_DIR / "sample_dataset_balanced.csv"
OUT_PATH = DATA_DIR / "sample_dataset_balanced_fixed.csv"

DESIRED_PRODUCTS = [
    "FD", "Short-Term-Debt-MF", "Debt-MF", "Hybrid-Conservative",
    "Hybrid-Aggressive", "Large-Cap-Equity", "Multi-Cap-Equity",
    "Mid-Small-Cap-Equity", "Gold-ETF", "Liquid-Fund"
]

def synth_for_product(product, n, seed=42, template_row=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # If we have a template row we can jitter around it; otherwise use random sensible defaults
        if template_row is not None:
            base = template_row.copy()
            # jitter numeric fields slightly
            income = max(5000, float(base.get("income", 50000)) * float(1 + rng.normal(0, 0.08)))
            expenses = max(0, float(base.get("expenses", income * 0.5)) * float(1 + rng.normal(0, 0.08)))
            age = int(max(18, float(base.get("age", 35)) + rng.integers(-8, 9)))
            goal_amount = max(10000, float(base.get("goal_amount", income * 4)) * float(1 + rng.normal(0, 0.3)))
            duration_months = int(max(6, int(base.get("duration_months", 24)) + rng.integers(-12, 13)))
            dependents = int(max(0, int(base.get("dependents", 0)) + rng.integers(-1, 2)))
            employment = base.get("employment", rng.choice(["salaried","self-employed","business","student","retired"]))
            # pick risk based on product type heuristics
            if product in ["FD", "Short-Term-Debt-MF", "Liquid-Fund"]:
                risk = rng.choice(["low","medium"], p=[0.7,0.3])
            elif product in ["Large-Cap-Equity", "Multi-Cap-Equity", "Mid-Small-Cap-Equity", "Hybrid-Aggressive"]:
                risk = rng.choice(["medium","high"], p=[0.35,0.65])
            else:
                risk = rng.choice(["low","medium","high"], p=[0.3,0.5,0.2])
            financial_health = rng.choice(["poor","average","good","excellent"], p=[0.1,0.4,0.35,0.15])

            total_holdings = max(0.0, float(rng.integers(0, int(max(1, income * 3)))))
            eq = float(total_holdings * max(0, rng.normal(0.2, 0.15)))
            debt = float(total_holdings * max(0, rng.normal(0.15, 0.12)))
            hybrid = float(total_holdings * max(0, rng.normal(0.08, 0.07)))
            gold = float(total_holdings * max(0, rng.normal(0.05, 0.05)))
            cash = float(total_holdings * max(0, rng.normal(0.15, 0.12)))

        else:
            # no template available — generate plausible defaults
            if product in ["FD", "Short-Term-Debt-MF", "Liquid-Fund"]:
                income = float(rng.integers(15000, 90000))
                risk = rng.choice(["low","medium"], p=[0.6,0.4])
            elif product in ["Large-Cap-Equity","Multi-Cap-Equity","Mid-Small-Cap-Equity","Hybrid-Aggressive"]:
                income = float(rng.integers(40000, 300000))
                risk = rng.choice(["medium","high"], p=[0.4,0.6])
            else:
                income = float(rng.integers(20000, 200000))
                risk = rng.choice(["low","medium","high"], p=[0.3,0.5,0.2])

            expenses = float(rng.integers(int(income * 0.2), int(income * 0.8)))
            age = int(rng.integers(22, 60))
            goal_amount = float(rng.integers(20000, 3000000))
            duration_months = int(rng.integers(6, 240))
            dependents = int(rng.integers(0, 4))
            employment = rng.choice(["salaried","self-employed","business","student","retired"])
            financial_health = rng.choice(["poor","average","good","excellent"])
            total_holdings = float(rng.integers(0, int(max(1, income * 3))))
            eq = total_holdings * rng.uniform(0,0.5)
            debt = total_holdings * rng.uniform(0,0.4)
            hybrid = total_holdings * rng.uniform(0,0.3)
            gold = total_holdings * rng.uniform(0,0.2)
            cash = total_holdings * rng.uniform(0,0.3)

        savings = max(income - expenses, 0.0)
        savings_rate = savings / (income + 1)
        base_ret = 0.06 if risk == "low" else (0.09 if risk == "medium" else 0.12)
        expected_return = float(max(0.02, np.clip(base_ret + rng.normal(0, 0.01), 0.02, 0.25)))

        row = {
            "income": round(float(income),2),
            "expenses": round(float(expenses),2),
            "savings": round(float(savings),2),
            "savings_rate": round(float(savings_rate),4),
            "goal_amount": round(float(goal_amount),2),
            "duration_months": int(duration_months),
            "age": int(age),
            "dependents": int(dependents),
            "employment": employment,
            "risk_level": risk,
            "financial_health": financial_health,
            "equity": round(float(eq),2),
            "debt": round(float(debt),2),
            "hybrid": round(float(hybrid),2),
            "gold": round(float(gold),2),
            "cash": round(float(cash),2),
            "recommended_product": product,
            "expected_return": round(expected_return,4),
        }
        rows.append(row)
    return pd.DataFrame(rows)

def main(target_per_class=None):
    if not IN_PATH.exists():
        raise SystemExit(f"Input file not found: {IN_PATH}. Place your sample_dataset_balanced.csv in sample_data/")

    df = pd.read_csv(IN_PATH)
    counts = df["recommended_product"].value_counts()
    print("Existing product counts:\n", counts)
    present = set(counts.index.tolist())
    missing = [p for p in DESIRED_PRODUCTS if p not in present]
    print("Missing products:", missing)

    if target_per_class is None:
        # default to current maximum count across existing classes
        target = int(counts.max())
    else:
        target = int(target_per_class)

    print(f"Target rows per class: {target}")

    generated = []
    # For each missing product, generate rows. For existing but under target, also upsample a bit
    for p in DESIRED_PRODUCTS:
        cur = int(counts.get(p, 0))
        need = max(0, target - cur)
        if need == 0:
            continue
        # try to pick a template row from an existing similar product if possible
        template_row = None
        # heuristic: if p is an equity variant, pick Multi-Cap-Equity or Large-Cap-Equity template if present
        if p in ["Hybrid-Aggressive", "Mid-Small-Cap-Equity", "Gold-ETF"]:
            # look for similar templates
            for prefer in ["Multi-Cap-Equity", "Large-Cap-Equity", "Hybrid-Conservative", "Liquid-Fund"]:
                if prefer in present:
                    template_row = df[df["recommended_product"] == prefer].sample(1, random_state=42).iloc[0].to_dict()
                    break
        # synthesize
        print(f"Generating {need} rows for product {p} (template: {'yes' if template_row is not None else 'no'})")
        gen = synth_for_product(p, need, seed=42 + hash(p) % 1000, template_row=template_row)
        generated.append(gen)

    if generated:
        gen_df = pd.concat(generated, ignore_index=True)
        out_df = pd.concat([df, gen_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        out_df = df.copy()

    out_df.to_csv(OUT_PATH, index=False)
    print("Wrote balanced fixed CSV:", OUT_PATH, "rows:", len(out_df))
    print("New counts:\n", out_df["recommended_product"].value_counts())

if __name__ == "__main__":
    # default: target_per_class None -> use current max count for equalization
    main(target_per_class=None)
