# Credit Risk Score Prediction

Projeto de Machine Learning para previsão de inadimplência utilizando o dataset "Give Me Some Credit" do Kaggle.

## Objetivo

Instituições financeiras precisam equilibrar dois objetivos:

* Aprovar o máximo possível de clientes
* Minimizar perdas por inadimplência

Este projeto utiliza Machine Learning para estimar a probabilidade de inadimplência e gerar um score de risco para auxiliar decisões de crédito.

## Dataset

Give Me Some Credit (Kaggle)

## Tecnologias

* Python
* Pandas
* Scikit-Learn
* LightGBM
* SHAP
* Streamlit

## Modelos Avaliados

* Logistic Regression
* Random Forest
* LightGBM

## Resultados

| Modelo              | AUC-ROC |
| ------------------- | ------- |
| Logistic Regression | 0.825   |
| Random Forest       | 0.866   |
| LightGBM            | 0.868   |

Modelo vencedor: LightGBM

## Explicabilidade

O projeto utiliza SHAP para identificar os fatores mais importantes na decisão do modelo.

Principais variáveis:

* Utilização do limite de crédito
* Histórico de atrasos
* Idade
* Quantidade de linhas de crédito
* Relação dívida/renda

## Dashboard

O projeto possui uma interface em Streamlit para simulação de score de crédito em tempo real.


## MisaelJunior
(https://www.linkedin.com/in/misael-junior-b9b9a8346/)