import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

HERE = os.path.dirname(__file__)
DATA = os.path.join(os.path.dirname(HERE), "sample_data", "dhanmitra_dataset.csv")
OUT_DIR = HERE
CLASSIFIER_PATH = os.path.join(OUT_DIR, "clf.pkl")
REGRESSOR_PATH = os.path.join(OUT_DIR, "reg.pkl")
LABELS_PATH = os.path.join(OUT_DIR, "labels.pkl")

def main():
    df = pd.read_csv(DATA)
    # Encoders
    enc_risk = LabelEncoder()
    df["risk_encoded"] = enc_risk.fit_transform(df["risk_level"])
    enc_prod = LabelEncoder()
    df["product_class"] = enc_prod.fit_transform(df["recommended_product"])

    features = ["income", "expenses", "goal_amount", "duration_months", "risk_encoded"]
    X = df[features]
    y_class = df["product_class"]
    y_return = df["expected_return"]

    X_train, X_test, y_class_train, y_class_test, y_ret_train, y_ret_test = train_test_split(
        X, y_class, y_return, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_class_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_class_test, y_pred, target_names=enc_prod.classes_))

    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X_train, y_ret_train)
    y_ret_pred = reg.predict(X_test)
    print("\nReturn Prediction:")
    print("MSE:", mean_squared_error(y_ret_test, y_ret_pred))
    print("R2:", r2_score(y_ret_test, y_ret_pred))

    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(reg, REGRESSOR_PATH)
    joblib.dump({"risk": enc_risk, "product": enc_prod}, LABELS_PATH)
    print(f"Saved models to {OUT_DIR}")

if __name__ == "__main__":
    main()
