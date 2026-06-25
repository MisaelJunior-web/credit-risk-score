from src.data_prep import run
from src.explain import global_importance

df = run()

X = df.drop(columns=["SeriousDlqin2yrs"])

global_importance(
    X.sample(min(1000, len(X)), random_state=42)
)