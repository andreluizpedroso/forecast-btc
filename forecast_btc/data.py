"""Acesso às fontes de dados usadas pelo projeto."""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATABASE_PATH


def load_price_history(
    ticker: str,
    database_path: str | Path = DATABASE_PATH,
) -> pd.DataFrame:
    """Carrega o histórico de fechamento de um ativo do SQLite."""
    query = """
        SELECT data AS Date, fechamento AS Close
        FROM historico
        WHERE ativo = ?
        ORDER BY data
    """

    with closing(sqlite3.connect(database_path)) as connection:
        history = pd.read_sql_query(query, connection, params=(ticker,))

    if history.empty:
        raise ValueError(f"Nenhum histórico encontrado para {ticker}.")

    history["Date"] = pd.to_datetime(history["Date"])
    history["Close"] = pd.to_numeric(history["Close"], errors="raise")
    return history


def download_recent_closes(ticker: str, period: str = "90d") -> np.ndarray:
    """Baixa fechamentos recentes do Yahoo Finance."""
    import yfinance as yf

    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if data.empty or "Close" not in data:
        raise ValueError(f"O Yahoo Finance não retornou dados para {ticker}.")

    closes = np.asarray(data["Close"], dtype=float).reshape(-1)
    return closes[np.isfinite(closes)]
