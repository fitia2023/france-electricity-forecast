import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> pd.DataFrame:
    """Calcule les métriques standards de prévision sur le jeu de test."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100

    return pd.DataFrame(
        {"MAE": [mae], "RMSE": [rmse], "MAPE (%)": [mape]},
        index=[model_name],
    )
