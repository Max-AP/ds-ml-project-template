"""Script de entrenamiento y evaluación del modelo final."""

from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

TARGET_COL = "median_house_value"
DEFAULT_MODEL_PARAMS = {
    "random_state": 42,
    "n_jobs": -1,
    "max_depth": None,
    "max_features": 0.6,
    "min_samples_leaf": 2,
    "min_samples_split": 4,
    "n_estimators": 800,
}


def _load_xy(processed_data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carga un dataset procesado y separa X/y."""
    data = pd.read_csv(processed_data_path)

    if TARGET_COL not in data.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no existe en {processed_data_path}")

    x_data = data.drop(columns=[TARGET_COL])
    y_data = data[TARGET_COL].copy()
    return x_data, y_data


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_best_model(
    processed_train_data_path: str,
    model_save_path: str,
    model_params: dict | None = None,
) -> RandomForestRegressor:
    """Entrena el mejor modelo sobre datos procesados y guarda el artefacto."""
    x_train, y_train = _load_xy(processed_train_data_path)

    params = DEFAULT_MODEL_PARAMS.copy()
    if model_params is not None:
        params.update(model_params)

    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)

    train_rmse = _rmse(y_train, model.predict(x_train))
    train_target_median = float(y_train.median())

    output_path = Path(model_save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "target_col": TARGET_COL,
        "params": params,
        "train_rmse": train_rmse,
        "train_target_median": train_target_median,
    }
    joblib.dump(payload, output_path)

    print(f"Modelo entrenado y guardado en: {output_path}")
    print(f"Parámetros usados: {params}")
    print(f"RMSE Train: {train_rmse:,.2f}")

    return model


def evaluate_model(model_path: str, processed_test_data_path: str) -> float:
    """Evalúa el modelo guardado sobre test y reporta RMSE + baseline."""
    payload = joblib.load(model_path)
    train_target_median: float | None
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        median_value = payload.get("train_target_median")
        train_target_median = float(median_value) if isinstance(median_value, Real) else None
    else:
        model = payload
        train_target_median = None

    x_test, y_test = _load_xy(processed_test_data_path)
    y_pred = model.predict(x_test)

    test_rmse = _rmse(y_test, y_pred)
    print(f"RMSE Test: {test_rmse:,.2f}")

    if train_target_median is not None:
        baseline_pred = np.full(shape=len(y_test), fill_value=train_target_median, dtype=float)
        baseline_rmse = _rmse(y_test, baseline_pred)
        improvement = (baseline_rmse - test_rmse) / baseline_rmse

        print(f"RMSE Test (baseline mediana train): {baseline_rmse:,.2f}")
        print(f"Mejora vs baseline: {improvement:.2%}")

    return test_rmse


if __name__ == "__main__":
    PROCESSED_TRAIN_PATH = "data/processed/train_processed.csv"
    PROCESSED_TEST_PATH = "data/processed/test_processed.csv"
    MODEL_OUTPUT_PATH = "models/best_model.pkl"

    train_best_model(PROCESSED_TRAIN_PATH, MODEL_OUTPUT_PATH)
    evaluate_model(MODEL_OUTPUT_PATH, PROCESSED_TEST_PATH)
