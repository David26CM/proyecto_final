# Proyecto Final - Predicción Bank Marketing

Trabajo final para el curso de MLOps. Consiste en entrenar un modelo con el dataset de Bank Marketing, exponerlo mediante una API con FastAPI y una interfaz con Gradio, además de pruebas automáticas con pytest.

## Estructura

- `app/`: Contiene la API (`api.py`) y la interfaz (`gradio_app.py`).
- `data/raw/`: Datos del dataset.
- `models/`: Modelo guardado (`bank_model.joblib`) y archivo de métricas.
- `src/final_mlops/`: Scripts de preparación de datos, entrenamiento y predicción.
- `tests/`: Pruebas del proyecto.

## Cómo ejecutar

Tener instalado Python y `uv`.

1. Instalar librerías:
```bash
uv sync --all-groups
uv pip install -e .
```

2. Entrenar el modelo:
```bash
uv run python -m final_mlops.train
```

3. Levantar la API:
```bash
uv run uvicorn app.api:app --reload --port 8000
```
La documentación queda en: `http://localhost:8000/docs`

4. Levantar la interfaz web:
```bash
uv run python app/gradio_app.py
```

5. Correr las pruebas:
```bash
uv run pytest tests/ -v
```