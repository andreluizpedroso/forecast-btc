"""Ponto de entrada ASGI mantido para `uvicorn main:app`."""

from forecast_btc.api import app

__all__ = ["app"]
