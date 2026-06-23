"""Interface Streamlit do Forecast BTC."""

import pandas as pd
import streamlit as st

from forecast_btc.charts import create_forecast_chart
from forecast_btc.config import (
    BITCOIN_IMAGE_PATH,
    DEFAULT_TICKER,
    MAX_FORECAST_DAYS,
)
from forecast_btc.data import load_price_history
from forecast_btc.forecasting import (
    PriceForecaster,
    build_forecast_frame,
    moving_average_recommendation,
)
from forecast_btc.model import get_sequence_length, load_trained_model


@st.cache_data
def get_history() -> pd.DataFrame:
    return load_price_history(DEFAULT_TICKER)


@st.cache_resource
def get_forecaster() -> PriceForecaster:
    model = load_trained_model()
    return PriceForecaster(model, get_sequence_length(model))


def render_app() -> None:
    st.set_page_config(page_title="Previsão Financeira Bitcoin", layout="wide")
    st.title("📊 Previsão do Preço Futuro do Bitcoin")

    _, image_column, _ = st.columns([1, 2, 1])
    with image_column:
        st.image(str(BITCOIN_IMAGE_PATH), use_container_width=True)

    try:
        history = get_history()
        forecaster = get_forecaster()
    except Exception as error:
        st.error(f"Não foi possível iniciar a aplicação: {error}")
        st.stop()

    last_known_date = pd.Timestamp(history["Date"].iloc[-1])
    available_dates = pd.date_range(
        start=last_known_date + pd.Timedelta(days=1),
        periods=MAX_FORECAST_DAYS,
        freq="D",
    ).date

    st.sidebar.header("Configuração da previsão")
    target_date = st.sidebar.select_slider(
        "Escolha uma data futura:",
        options=available_dates,
        value=available_dates[0],
    )

    if not st.sidebar.button("Gerar previsão", type="primary"):
        return

    try:
        days = (pd.Timestamp(target_date) - last_known_date.normalize()).days
        predictions = forecaster.predict(history["Close"], days)
        forecast_data = build_forecast_frame(history, predictions)
        recommendation = moving_average_recommendation(forecast_data, target_date)

        st.subheader(f"Resultado para {target_date:%d/%m/%Y}")
        st.metric("Preço previsto", f"US$ {predictions[-1]:,.2f}")
        st.success(recommendation)
        st.plotly_chart(
            create_forecast_chart(forecast_data, target_date),
            use_container_width=True,
        )
        st.caption(
            "Estimativa experimental baseada apenas no histórico; "
            "não constitui recomendação financeira."
        )
    except Exception as error:
        st.error(f"Não foi possível gerar a previsão: {error}")


render_app()
