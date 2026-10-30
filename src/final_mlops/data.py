import pandas as pd
from sklearn.model_selection import train_test_split
from final_mlops.config import (
    DATA_PATH,
    TARGET,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, sep=";")

def validate_data(df):
    cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Falta columna: {col}")

    if df[cols].isna().any().any():
        raise ValueError("Hay valores nulos en el dataset")

    if (df["age"] <= 0).any() or (df["duration"] < 0).any():
        raise ValueError("Valores fuera de rango en age o duration")

def get_data():
    df = load_data()
    validate_data(df)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == "yes").astype(int)

    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

def main():
    X_train, X_test, y_train, y_test = get_data()
    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

if __name__ == "__main__":
    main()