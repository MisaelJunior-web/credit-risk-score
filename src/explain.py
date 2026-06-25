"""
explain.py
Explicabilidade do modelo com SHAP — gera o gráfico de importância global
de features e permite explicar uma predição individual.
"""

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path("models/model.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")
FIGURES_DIR = Path("reports/figures")


def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names


def global_importance(X_sample: pd.DataFrame, save: bool = True):
    model, _ = load_model_and_features()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
        print(f"Gráfico salvo em {FIGURES_DIR / 'shap_summary.png'}")
    plt.close()


def explain_single_prediction(client_features: dict) -> dict:
    """
    Retorna a contribuição de cada feature para a predição de um cliente
    específico, útil para a API explicar o score retornado.
    """
    model, feature_names = load_model_and_features()
    X = pd.DataFrame([client_features])[feature_names]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Para modelos binários TreeExplainer retorna lista [classe0, classe1]
    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    contributions = dict(zip(feature_names, values))
    sorted_contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    return sorted_contributions


if __name__ == "__main__":
    print("Use este módulo importado a partir de um notebook ou da API,")
    print("passando uma amostra de X (DataFrame) ou um dict de features de um cliente.")
