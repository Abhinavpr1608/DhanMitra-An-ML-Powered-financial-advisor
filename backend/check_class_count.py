# check_class_counts.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_dataset.csv"

df = pd.read_csv(DATA_PATH)
print("Total rows:", len(df))
print("\nClass distribution (recommended_product):")
print(df['recommended_product'].value_counts(dropna=False))
print("\nSample rows per class:")
print(df.groupby('recommended_product').size().sort_values(ascending=False).head(20))
