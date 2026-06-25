"""
streamlit_app.py
Interface simples para simular o score de crédito de um cliente, consumindo
a API FastAPI (ou o modelo direto, caso a API não esteja disponível).

Rodar com: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import pandas as pd
import streamlit as st

from score import probability_to_score, score_to_risk_band

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "models" / "feature_names.pkl"

st.set_page_config(page_title="Credit Risk Score", page_icon="💳", layout="centered")
st.title("💳 Simulador de Score de Crédito")
st.write(
    "Preencha os dados abaixo para simular a probabilidade de inadimplência "
    "e o score de risco (0–1000) de um cliente."
)

with st.form("client_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Idade", min_value=18, max_value=100, value=35)
        monthly_income = st.number_input("Renda mensal (R$)", min_value=0.0, value=4000.0)
        revolving_util = st.slider("Uso do limite de crédito disponível", 0.0, 2.0, 0.3)
        debt_ratio = st.slider("Razão dívida/renda", 0.0, 2.0, 0.4)
        dependents = st.number_input("Número de dependentes", min_value=0, value=0)
    with col2:
        open_credit_lines = st.number_input("Linhas de crédito abertas", min_value=0, value=5)
        real_estate_loans = st.number_input("Empréstimos imobiliários", min_value=0, value=1)
        past_due_30_59 = st.number_input("Atrasos de 30-59 dias", min_value=0, value=0)
        past_due_60_89 = st.number_input("Atrasos de 60-89 dias", min_value=0, value=0)
        past_due_90 = st.number_input("Atrasos de 90+ dias", min_value=0, value=0)

    submitted = st.form_submit_button("Calcular score")

if submitted:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    safe_income = monthly_income or 1
    features = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": past_due_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
        "NumberOfDependents": dependents,
    }
    features["debt_to_income"] = debt_ratio * safe_income
    features["total_past_due"] = past_due_30_59 + past_due_60_89 + past_due_90
    features["credit_lines_per_dependent"] = open_credit_lines / (dependents + 1)
    features["income_per_dependent"] = monthly_income / (dependents + 1)

    X = pd.DataFrame([features])[feature_names]
    proba = model.predict_proba(X)[0, 1]
    score = probability_to_score(proba)
    band = score_to_risk_band(score)

    st.divider()
    st.metric("Score de crédito", f"{score} / 1000")
    st.metric("Probabilidade de inadimplência", f"{proba:.1%}")

    if band == "Baixo risco":
        st.success(f"Classificação: {band}")
    elif band == "Risco moderado":
        st.warning(f"Classificação: {band}")
    else:
        st.error(f"Classificação: {band}")
