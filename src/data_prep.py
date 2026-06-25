"""
data_prep.py
Limpeza e feature engineering do dataset "Give Me Some Credit".

Dataset esperado em data/raw/cs-training.csv (baixar do Kaggle):
https://www.kaggle.com/c/GiveMeSomeCredit

Colunas originais (resumo):
- SeriousDlqin2yrs: target (1 = inadimplente em até 2 anos)
- RevolvingUtilizationOfUnsecuredLines: uso do limite de crédito disponível
- age: idade
- NumberOfTime30-59DaysPastDueNotWorse: nº de atrasos 30-59 dias
- DebtRatio: razão dívida/renda
- MonthlyIncome: renda mensal
- NumberOfOpenCreditLinesAndLoans: nº de linhas de crédito abertas
- NumberOfTimes90DaysLate: nº de atrasos 90+ dias
- NumberRealEstateLoansOrLines: nº de empréstimos imobiliários
- NumberOfTime60-89DaysPastDueNotWorse: nº de atrasos 60-89 dias
- NumberOfDependents: nº de dependentes
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/cs-training.csv")
PROCESSED_PATH = Path("data/processed/credit_features.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MonthlyIncome e NumberOfDependents têm muitos nulos -> imputar pela mediana
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Outliers conhecidos do dataset: age=0 e valores absurdos de atraso (96, 98)
    df = df[df["age"] > 0]
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Renda mensal pode ser 0 -> evitar divisão por zero
    safe_income = df["MonthlyIncome"].replace(0, np.nan)

    df["debt_to_income"] = df["DebtRatio"] * safe_income
    df["debt_to_income"] = df["debt_to_income"].fillna(df["debt_to_income"].median())

    df["total_past_due"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    df["credit_lines_per_dependent"] = df["NumberOfOpenCreditLinesAndLoans"] / (
        df["NumberOfDependents"] + 1
    )

    df["income_per_dependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    return df


def run(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Dados processados salvos em {out_path} ({len(df)} linhas)")
    return df


if __name__ == "__main__":
    run()
