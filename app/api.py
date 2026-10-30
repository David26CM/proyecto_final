import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from final_mlops.config import MODEL_PATH

app = FastAPI(title="Bank Marketing Prediction API", version="0.1.0")


class ClientData(BaseModel):
    age: int = Field(..., ge=18, le=120, example=35)
    balance: float = Field(..., example=1500.0)
    duration: float = Field(..., ge=0, example=120.0)
    campaign: int = Field(..., ge=1, example=1)
    previous: int = Field(..., ge=0, example=0)
    job: str = Field(..., example="technician")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="tertiary")
    housing: str = Field(..., example="no")
    loan: str = Field(..., example="no")


def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(data: ClientData):
    try:
        model = get_model()
        df = pd.DataFrame([data.model_dump()])
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        return {
            "prediction": prediction,
            "deposit_subscribed": "yes" if prediction == 1 else "no",
            "probability": round(probability, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

