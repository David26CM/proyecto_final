from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "bank-full.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "bank_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Datos
NUMERICAL_FEATURES = [
    "age",
    "balance",
    "duration",
    "campaign",
    "previous"
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "housing",
    "loan"
]

TARGET = "y"

# Entrenamiento
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Calidad mínima
MIN_F1 = 0.40