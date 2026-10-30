import gradio as gr
import joblib
import pandas as pd
from final_mlops.config import MODEL_PATH

model = joblib.load(MODEL_PATH)


def predict_subscription(age, balance, duration, campaign, previous, job, marital, education, housing, loan):
    data = {
        "age": [age],
        "balance": [balance],
        "duration": [duration],
        "campaign": [campaign],
        "previous": [previous],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "housing": [housing],
        "loan": [loan],
    }
    df = pd.DataFrame(data)
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    resultado = "Suscrito (Sí)" if pred == 1 else "No suscrito (No)"
    return f"{resultado} | Probabilidad estimada: {prob:.2%}"


demo = gr.Interface(
    fn=predict_subscription,
    inputs=[
        gr.Number(label="Edad", value=35),
        gr.Number(label="Balance (€)", value=1500),
        gr.Number(label="Duración del contacto (seg)", value=180),
        gr.Number(label="Contactos en esta campaña", value=1),
        gr.Number(label="Contactos previos", value=0),
        gr.Dropdown(
            choices=[
                "admin.",
                "blue-collar",
                "technician",
                "services",
                "management",
                "retired",
                "student",
                "self-employed",
                "housemaid",
                "entrepreneur",
                "unemployed",
                "unknown",
            ],
            label="Ocupación",
            value="technician",
        ),
        gr.Dropdown(choices=["married", "single", "divorced"], label="Estado civil", value="married"),
        gr.Dropdown(
            choices=["primary", "secondary", "tertiary", "unknown"],
            label="Nivel educativo",
            value="tertiary",
        ),
        gr.Dropdown(choices=["yes", "no"], label="Préstamo hipotecario", value="no"),
        gr.Dropdown(choices=["yes", "no"], label="Préstamo personal", value="no"),
    ],
    outputs=gr.Textbox(label="Resultado de Inferencia"),
    title="Predicción de Suscripción a Depósito - Bank Marketing",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)