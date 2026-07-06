"""Carregamento e inspeção do modelo de Machine Learning."""

from pathlib import Path
from typing import Any

from .config import MODEL_PATH


def load_trained_model(model_path: str | Path = MODEL_PATH) -> Any:
    """Carrega o modelo persistido e apresenta um erro claro se ele não existir."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")

    from tensorflow.keras.models import load_model

    return load_model(path, compile=False)


def get_sequence_length(model: Any) -> int:
    """Obtém a janela temporal esperada pela entrada do modelo."""
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 3 or input_shape[1] is None:
        raise ValueError(f"Formato de entrada inesperado: {input_shape}")

    return int(input_shape[1])
