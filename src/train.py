"""
train.py
Treina e compara Logistic Regression, Random Forest e LightGBM para prever
SeriousDlqin2yrs. Salva o melhor modelo em models/model.pkl.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, classification_report
import lightgbm as lgb

from data_prep import run as prepare_data, PROCESSED_PATH

TARGET = "SeriousDlqin2yrs"
MODEL_PATH = Path("models/model.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")


def load_dataset() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        return prepare_data()
    return pd.read_csv(PROCESSED_PATH)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(name: str, y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "model": name,
        "auc_roc": roc_auc_score(y_true, y_proba),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }
    print(f"\n--- {name} ---")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Recall:  {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(classification_report(y_true, y_pred))
    return metrics


def train_logistic(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:, 1]
    return model, scaler, proba


def train_random_forest(X_train, y_train, X_test):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


def train_lightgbm(X_train, y_train, X_test):
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    results = []

    log_model, scaler, proba_log = train_logistic(X_train, y_train, X_test)
    results.append(evaluate("Logistic Regression", y_test, proba_log))

    rf_model, proba_rf = train_random_forest(X_train, y_train, X_test)
    results.append(evaluate("Random Forest", y_test, proba_rf))

    lgb_model, proba_lgb = train_lightgbm(X_train, y_train, X_test)
    results.append(evaluate("LightGBM", y_test, proba_lgb))

    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    print("\n=== Comparativo final ===")
    print(results_df.to_string(index=False))

    # Seleciona o melhor modelo por AUC-ROC (em produção, considerar recall + custo de negócio)
    best_name = results_df.iloc[0]["model"]
    best_model = {"Logistic Regression": log_model, "Random Forest": rf_model, "LightGBM": lgb_model}[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(list(X_train.columns), FEATURES_PATH)
    print(f"\nMelhor modelo ({best_name}) salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
