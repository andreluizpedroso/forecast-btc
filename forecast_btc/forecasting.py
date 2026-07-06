"""Regras de previsão e recomendação, independentes das interfaces."""

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import LONG_MOVING_AVERAGE, SHORT_MOVING_AVERAGE


class PriceForecaster:
    """Executa previsões recursivas com um modelo LSTM já treinado."""

    def __init__(self, model: Any, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("A janela temporal deve ser positiva.")
        self.model = model
        self.sequence_length = sequence_length

    def predict(self, prices: Iterable[float], days: int = 1) -> np.ndarray:
        """Prevê um ou mais dias usando cada saída como entrada do próximo passo."""
        values = np.asarray(list(prices), dtype=float).reshape(-1, 1)
        self._validate_input(values, days)

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        window = scaled_values[-self.sequence_length :].reshape(-1)
        predictions: list[float] = []

        for _ in range(days):
            model_input = window[-self.sequence_length :].reshape(
                1, self.sequence_length, 1
            )
            predicted = float(
                np.asarray(self.model.predict(model_input, verbose=0)).reshape(-1)[0]
            )
            predictions.append(predicted)
            window = np.append(window, predicted)

        return scaler.inverse_transform(
            np.asarray(predictions).reshape(-1, 1)
        ).reshape(-1)

    def predict_to_date(
        self,
        prices: Iterable[float],
        last_known_date: object,
        target_date: object,
    ) -> float:
        """Prevê o fechamento de uma data posterior ao último dado conhecido."""
        last_date = pd.Timestamp(last_known_date).normalize()
        target = pd.Timestamp(target_date).normalize()
        days = (target - last_date).days
        if days <= 0:
            raise ValueError("A data da previsão deve ser futura.")
        return float(self.predict(prices, days)[-1])

    def _validate_input(self, values: np.ndarray, days: int) -> None:
        if days <= 0:
            raise ValueError("A quantidade de dias deve ser positiva.")
        if len(values) < self.sequence_length:
            raise ValueError(
                f"São necessários ao menos {self.sequence_length} preços históricos."
            )
        if not np.isfinite(values).all():
            raise ValueError("O histórico contém preços inválidos.")


def build_forecast_frame(
    history: pd.DataFrame,
    predictions: Iterable[float],
) -> pd.DataFrame:
    """Combina histórico, previsão e médias móveis em uma única tabela."""
    required_columns = {"Date", "Close"}
    if not required_columns.issubset(history.columns):
        raise ValueError("O histórico deve conter as colunas Date e Close.")

    forecast_values = np.asarray(list(predictions), dtype=float)
    last_date = pd.Timestamp(history["Date"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(forecast_values),
        freq="D",
    )
    forecast = pd.DataFrame({"Date": future_dates, "Close": forecast_values})
    full_data = pd.concat(
        [history[["Date", "Close"]].copy(), forecast], ignore_index=True
    )
    full_data["MM20"] = full_data["Close"].rolling(SHORT_MOVING_AVERAGE).mean()
    full_data["MM80"] = full_data["Close"].rolling(LONG_MOVING_AVERAGE).mean()
    return full_data


def moving_average_recommendation(
    forecast_data: pd.DataFrame,
    target_date: object,
) -> str:
    """Produz um sinal simples com base no cruzamento das médias móveis."""
    target = pd.Timestamp(target_date).normalize()
    dates = pd.to_datetime(forecast_data["Date"]).dt.normalize()
    target_rows = forecast_data.loc[dates == target]

    if target_rows.empty:
        raise ValueError("A data escolhida não está no período previsto.")

    mm20 = float(target_rows["MM20"].iloc[0])
    mm80 = float(target_rows["MM80"].iloc[0])
    if np.isnan(mm20) or np.isnan(mm80):
        return "Não há dados suficientes para gerar uma recomendação."
    if mm20 > mm80:
        return "✅ Recomendação: Comprar"
    return "⏳ Recomendação: Aguardar"
