"""
api.py
API FastAPI que recebe os dados de um cliente e retorna o score de risco
de crédito (0-1000) junto com os principais fatores que influenciaram a decisão.

Rodar com: uvicorn app.api:app --reload
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from score import probability_to_score, score_to_risk_band
from explain import explain_single_prediction

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "models" / "feature_names.pkl"

app = FastAPI(
    title="Credit Risk Score API",
    description="API de previsão de inadimplência e score de risco de crédito",
    version="1.0.0",
)

model = None
feature_names = None


@app.on_event("startup")
def load_model():
    global model, feature_names
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)


class ClientData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0)
    age: int = Field(..., gt=0)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: float = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int = Field(..., ge=0)

    class Config:
        populate_by_name = True


class ScoreResponse(BaseModel):
    probability_of_default: float
    score: int
    risk_band: str
    top_factors: dict


def build_features(client: ClientData) -> dict:
    data = client.dict(by_alias=True)
    safe_income = data["MonthlyIncome"] or 1
    data["debt_to_income"] = data["DebtRatio"] * safe_income
    data["total_past_due"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + data["NumberOfTime60-89DaysPastDueNotWorse"]
        + data["NumberOfTimes90DaysLate"]
    )
    data["credit_lines_per_dependent"] = data["NumberOfOpenCreditLinesAndLoans"] / (
        data["NumberOfDependents"] + 1
    )
    data["income_per_dependent"] = data["MonthlyIncome"] / (data["NumberOfDependents"] + 1)
    return data


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Credit Risk Score API rodando"}


@app.post("/predict", response_model=ScoreResponse)
def predict(client: ClientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    features = build_features(client)
    X = pd.DataFrame([features])[feature_names]

    proba = model.predict_proba(X)[0, 1]
    score = probability_to_score(proba)
    band = score_to_risk_band(score)

    try:
        contributions = explain_single_prediction(features)
        top_factors = dict(list(contributions.items())[:5])
    except Exception:
        top_factors = {}

    return ScoreResponse(
        probability_of_default=round(float(proba), 4),
        score=score,
        risk_band=band,
        top_factors=top_factors,
    )
