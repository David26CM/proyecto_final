import json
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from final_mlops.config import (
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MIN_F1,
    MODEL_PATH,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
)
from final_mlops.data import get_data


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train():
    X_train, X_test, y_train, y_test = get_data()

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "threshold_f1": MIN_F1,
        "approved": bool(f1 >= MIN_F1),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas obtenidas: Accuracy = {metrics['accuracy']} | F1-Score = {metrics['f1_score']}")

    # Puerta de calidad
    if not metrics["approved"]:
        raise ValueError(f"Modelo rechazado: F1 {f1:.4f} menor al umbral {MIN_F1}")

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo aprobado y guardado en: {MODEL_PATH}")


def main():
    train()


if __name__ == "__main__":
    main()