"""Visualizações da aplicação."""

import pandas as pd
import plotly.graph_objects as go


def create_forecast_chart(data: pd.DataFrame, target_date: object) -> go.Figure:
    """Cria o gráfico de histórico, previsão e médias móveis."""
    target = pd.Timestamp(target_date)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Fechamento")
    )
    figure.add_trace(
        go.Scatter(x=data["Date"], y=data["MM20"], mode="lines", name="MM20")
    )
    figure.add_trace(
        go.Scatter(x=data["Date"], y=data["MM80"], mode="lines", name="MM80")
    )
    figure.add_vline(line_color="red", line_dash="dash", x=target.timestamp() * 1000)
    figure.add_annotation(
        x=target,
        y=data["Close"].max(),
        text="Data escolhida",
        showarrow=True,
        arrowhead=1,
    )
    figure.update_layout(
        title="Histórico, previsão e médias móveis",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        template="plotly_white",
        height=600,
        xaxis={"type": "date"},
    )
    return figure
