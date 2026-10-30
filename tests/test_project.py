import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.api import app
from final_mlops.data import get_data, validate_data
from final_mlops.config import MODEL_PATH, DATA_PATH
import joblib

client = TestClient(app)


def _ensure_dummy_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        dummy_df = pd.DataFrame(
            {
                "age": [30, 40, 50, 35, 60, 25, 45, 55, 33, 48],
                "job": ["services"] * 10,
                "marital": ["married"] * 10,
                "education": ["secondary"] * 10,
                "default": ["no"] * 10,
                "balance": [1000] * 10,
                "housing": ["no"] * 10,
                "loan": ["no"] * 10,
                "contact": ["cellular"] * 10,
                "day": [1] * 10,
                "month": ["may"] * 10,
                "duration": [150] * 10,
                "campaign": [1] * 10,
                "pdays": [-1] * 10,
                "previous": [0] * 10,
                "poutcome": ["unknown"] * 10,
                "y": ["no", "yes", "no", "no", "yes", "no", "yes", "no", "no", "yes"],
            }
        )
        dummy_df.to_csv(DATA_PATH, sep=";", index=False)


def test_data_pipeline():
    _ensure_dummy_data()
    X_train, X_test, y_train, y_test = get_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) > len(X_test)


def test_data_validation_errors():
    df_invalid = pd.DataFrame({"age": [-5], "duration": [-10]})
    with pytest.raises(ValueError):
        validate_data(df_invalid)


def test_model_artifact_exists():
    assert MODEL_PATH.exists()
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")


def test_api_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_predict():
    payload = {
        "age": 35,
        "balance": 1500.0,
        "duration": 180.0,
        "campaign": 1,
        "previous": 0,
        "job": "technician",
        "marital": "married",
        "education": "tertiary",
        "housing": "no",
        "loan": "no",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data