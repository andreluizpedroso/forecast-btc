import sqlite3

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

SEQUENCE_LENGTH = 10
TICKER = "BTC-USD"

app = FastAPI(title="API de Previsão do Bitcoin com LSTM")

model = load_model("modelo_lstm.h5", compile=False)


def _load_scaler() -> MinMaxScaler:
    conn = sqlite3.connect("finance.db")
    df = pd.read_sql_query(
        "SELECT fechamento FROM historico WHERE ativo = ? ORDER BY data",
        conn, params=(TICKER,)
    )
    conn.close()
    train_size = int(len(df) * 0.7)
    scaler = MinMaxScaler()
    scaler.fit(df[["fechamento"]].iloc[:train_size])
    return scaler


scaler = _load_scaler()


class PrevisaoRequest(BaseModel):
    historico: list[float]


@app.get("/")
def home():
    return {"mensagem": "API de previsão do Bitcoin com LSTM está online."}


@app.post("/prever")
def prever(request: PrevisaoRequest):
    if len(request.historico) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"São necessários ao menos {SEQUENCE_LENGTH} preços históricos.",
        )
    try:
        dados_array = np.array(request.historico[-SEQUENCE_LENGTH:]).reshape(-1, 1)
        dados_normalizados = scaler.transform(dados_array)
        entrada = dados_normalizados.reshape(1, SEQUENCE_LENGTH, 1)
        previsao_normalizada = model.predict(entrada)
        previsao = scaler.inverse_transform(previsao_normalizada)
        return {"preco_previsto": float(previsao[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
