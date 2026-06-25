"""
score.py
Converte a probabilidade de inadimplência prevista pelo modelo em um score
de risco na escala 0-1000, no estilo dos scores de crédito de mercado.

Convenção adotada: score ALTO = risco BAIXO (cliente bom pagador), igual à
maioria dos scores comerciais (Serasa, Boa Vista etc).
"""

import numpy as np


def probability_to_score(proba: float, min_score: int = 0, max_score: int = 1000) -> int:
    """
    Transforma probabilidade de inadimplência (0 a 1) em score de risco (0 a 1000).
    proba=0 (sem risco) -> score 1000
    proba=1 (risco máximo) -> score 0
    """
    proba = np.clip(proba, 0.0, 1.0)
    score = max_score - (proba * (max_score - min_score))
    return int(round(score))


def score_to_risk_band(score: int) -> str:
    if score >= 800:
        return "Baixo risco"
    elif score >= 600:
        return "Risco moderado"
    elif score >= 400:
        return "Risco alto"
    else:
        return "Risco muito alto"


if __name__ == "__main__":
    for p in [0.01, 0.1, 0.3, 0.5, 0.8, 0.95]:
        s = probability_to_score(p)
        print(f"Probabilidade={p:.2f} -> Score={s} ({score_to_risk_band(s)})")
