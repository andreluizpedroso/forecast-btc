"""API HTTP do Forecast BTC."""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .config import DEFAULT_TICKER
from .data import download_recent_closes
from .forecasting import PriceForecaster
from .model import get_sequence_length, load_trained_model


class PredictionRequest(BaseModel):
    historico: list[float]


class PredictionResponse(BaseModel):
    preco_previsto: float
    dias_usados: int
    fonte: str | None = None


@lru_cache(maxsize=1)
def get_forecaster() -> PriceForecaster:
    model = load_trained_model()
    return PriceForecaster(model, get_sequence_length(model))


def create_app() -> FastAPI:
    application = FastAPI(
        title="API de Previsão do Bitcoin com LSTM",
        version="1.0.0",
    )

    @application.get("/")
    def health_check() -> dict[str, str]:
        return {"mensagem": "API de previsão do Bitcoin está online."}

    @application.post("/prever", response_model=PredictionResponse)
    def predict_manual(
        request: PredictionRequest,
        forecaster: PriceForecaster = Depends(get_forecaster),
    ) -> PredictionResponse:
        try:
            prediction = forecaster.predict(request.historico)[0]
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return PredictionResponse(
            preco_previsto=float(prediction),
            dias_usados=forecaster.sequence_length,
            fonte="Entrada manual",
        )

    @application.post("/prever_auto", response_model=PredictionResponse)
    def predict_automatic(
        forecaster: PriceForecaster = Depends(get_forecaster),
    ) -> PredictionResponse:
        try:
            closes = download_recent_closes(DEFAULT_TICKER)
            prediction = forecaster.predict(closes)[0]
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Não foi possível consultar os dados de mercado.",
            ) from error

        return PredictionResponse(
            preco_previsto=float(prediction),
            dias_usados=forecaster.sequence_length,
            fonte="Yahoo Finance",
        )

    return application


app = create_app()
