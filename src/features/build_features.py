"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

TARGET_COL = "median_house_value"
CATEGORICAL_COL = "ocean_proximity"
COUNT_COLS = ["total_rooms", "total_bedrooms", "population", "households"]
IQR_COLS = [
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "housing_median_age",
]


def _iqr_clip(series: pd.Series) -> pd.Series:
    """Acota outliers usando el criterio IQR (winsorización simple)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def _scale_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = False,
    expected_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """Escala columnas numéricas (excepto dummies) y preserva el target."""
    data = df.copy()

    y = None
    if TARGET_COL in data.columns:
        y = data[TARGET_COL].copy()
        data = data.drop(columns=[TARGET_COL])

    if expected_feature_columns is not None:
        data = data.reindex(columns=expected_feature_columns, fill_value=0)

    dummy_cols = [col for col in data.columns if col.startswith("ocean_")]
    numeric_features = data.select_dtypes(include="number").columns.difference(dummy_cols)

    scaler_to_use = scaler if scaler is not None else StandardScaler()
    if len(numeric_features) > 0:
        if fit:
            data[numeric_features] = scaler_to_use.fit_transform(data[numeric_features])
        else:
            data[numeric_features] = scaler_to_use.transform(data[numeric_features])

    if y is not None:
        result = data.copy()
        result[TARGET_COL] = y.values
    else:
        result = data

    return result, scaler_to_use, data.columns.tolist()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Maneja los valores faltantes.
       Puedes llenarlos con la mediana de la columna.
    2. Retorna el DataFrame limpio.
    """
    cleaned = df.copy()

    # Completitud: imputación de total_bedrooms con mediana
    if "total_bedrooms" in cleaned.columns:
        cleaned["total_bedrooms"] = cleaned["total_bedrooms"].fillna(cleaned["total_bedrooms"].median())

    # Consistencia: eliminar duplicados exactos
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Precisión: valores no válidos (<= 0) en columnas de conteo
    for col in COUNT_COLS:
        if col in cleaned.columns:
            invalid_mask = cleaned[col] <= 0
            if invalid_mask.any():
                cleaned.loc[invalid_mask, col] = np.nan
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Sensibilidad: clipping IQR para reducir impacto de outliers
    for col in IQR_COLS:
        if col in cleaned.columns:
            cleaned[col] = _iqr_clip(cleaned[col])

    return cleaned

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Agrega nuevas variables derivando las existentes, por ejemplo:
       - 'rooms_per_household' = total_rooms / households
       - 'population_per_household' = population / households
       - 'bedrooms_per_room' = total_bedrooms / total_rooms
    2. Retorna el DataFrame enriquecido.
    """
    featured = df.copy()

    if {"total_rooms", "households"}.issubset(featured.columns):
        featured["rooms_per_household"] = featured["total_rooms"] / featured["households"].replace(0, np.nan)

    if {"total_bedrooms", "total_rooms"}.issubset(featured.columns):
        featured["bedrooms_per_room"] = featured["total_bedrooms"] / featured["total_rooms"].replace(0, np.nan)

    if {"population", "households"}.issubset(featured.columns):
        featured["population_per_household"] = featured["population"] / featured["households"].replace(0, np.nan)

    created_cols = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    for col in created_cols:
        if col in featured.columns:
            featured[col] = featured[col].replace([np.inf, -np.inf], np.nan)
            featured[col] = featured[col].fillna(featured[col].median())

    return featured

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que toma el DataFrame crudo y aplica limpieza y enriquecimiento.
    """
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)

    if CATEGORICAL_COL in df_featured.columns:
        df_featured = pd.get_dummies(
            df_featured,
            columns=[CATEGORICAL_COL],
            prefix="ocean",
            dtype=int,
        )

    return df_featured


def prepare_train_test_sets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
    """Aplica el pipeline a train/test y escala de forma consistente sin fuga de datos."""
    train_prepared = preprocess_pipeline(train_df)
    test_prepared = preprocess_pipeline(test_df)

    train_scaled, scaler, feature_columns = _scale_features(train_prepared, fit=True)
    test_scaled, _, _ = _scale_features(
        test_prepared,
        scaler=scaler,
        fit=False,
        expected_feature_columns=feature_columns,
    )

    return train_scaled, test_scaled, scaler, feature_columns

if __name__ == "__main__":
    INTERIM_PATH = Path("data/interim")
    PROCESSED_PATH = Path("data/processed")
    MODELS_PATH = Path("models")

    train_set = pd.read_csv(INTERIM_PATH / "train_set.csv")
    test_set = pd.read_csv(INTERIM_PATH / "test_set.csv")

    train_processed, test_processed, scaler, feature_columns = prepare_train_test_sets(train_set, test_set)

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    train_processed.to_csv(PROCESSED_PATH / "train_processed.csv", index=False)
    test_processed.to_csv(PROCESSED_PATH / "test_processed.csv", index=False)

    joblib.dump(
        {"scaler": scaler, "feature_columns": feature_columns},
        MODELS_PATH / "preprocessor.pkl",
    )

    print("Datos procesados guardados en data/processed/ y preprocesador en models/preprocessor.pkl")
