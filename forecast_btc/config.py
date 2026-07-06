"""Configurações centrais e caminhos do projeto."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "finance.db"
MODEL_PATH = PROJECT_ROOT / "modelo_lstm.h5"
BITCOIN_IMAGE_PATH = PROJECT_ROOT / "bitcoin_img.png"

DEFAULT_TICKER = "BTC-USD"
MAX_FORECAST_DAYS = 30
SHORT_MOVING_AVERAGE = 20
LONG_MOVING_AVERAGE = 80
