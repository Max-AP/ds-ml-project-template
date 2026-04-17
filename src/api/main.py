"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.build_features import preprocess_pipeline

# INSTRUCCIONES: Define el esquema de datos esperado por la API (Las variables X que usa tu modelo)
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"

# Variable global para cargar el modelo
# IMPORTANTE: Asegúrate de guardar tu modelo en "models/best_model.pkl" o ajusta la ruta
model: Any = None
preprocessor_bundle: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Carga artefactos al iniciar la app."""
    load_model()
    yield


# Inicializamos la app
app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)",
    version="1.0",
    lifespan=lifespan,
)


def _extract_loaded_model(payload: Any) -> Any:
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _features_to_dict(features: HousingFeatures) -> dict[str, Any]:
    model_dump_method = getattr(features, "model_dump", None)
    if callable(model_dump_method):
        return model_dump_method()

    dict_method = getattr(features, "dict", None)
    if callable(dict_method):
        return dict_method()

    raise RuntimeError("No se pudo serializar el payload de entrada")


def _prepare_features_for_model(features: HousingFeatures) -> pd.DataFrame:
    if preprocessor_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="El preprocesador no está cargado. Genera models/preprocessor.pkl antes de usar la API.",
        )

    scaler = preprocessor_bundle.get("scaler")
    feature_columns = preprocessor_bundle.get("feature_columns")
    if scaler is None or not isinstance(feature_columns, list):
        raise HTTPException(status_code=500, detail="El preprocesador cargado es inválido.")

    input_df = pd.DataFrame([_features_to_dict(features)])
    prepared_df = preprocess_pipeline(input_df)
    prepared_df = prepared_df.reindex(columns=feature_columns, fill_value=0)

    dummy_cols = [col for col in prepared_df.columns if col.startswith("ocean_")]
    numeric_features = prepared_df.select_dtypes(include="number").columns.difference(dummy_cols)
    if len(numeric_features) > 0:
        prepared_df.loc[:, numeric_features] = scaler.transform(prepared_df[numeric_features])

    return prepared_df


def load_model():
    """
    Carga el modelo globalmente al iniciar el servidor usando joblib.
    """
    global model, preprocessor_bundle
    try:
        model_payload = joblib.load(MODEL_PATH)
        model = _extract_loaded_model(model_payload)
    except Exception:
        model = None
        print("Advertencia: No se pudo cargar el modelo. ¿Ya lo entrenaste y guardaste?")

    try:
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
        preprocessor_bundle = loaded_preprocessor if isinstance(loaded_preprocessor, dict) else None
    except Exception:
        preprocessor_bundle = None
        print("Advertencia: No se pudo cargar el preprocesador. Ejecuta src/features/build_features.py")

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos"}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    INSTRUCCIONES:
    1. Convierte el objeto 'features' (Pydantic) a un formato que Scikit-Learn entienda (ej un DataFrame o Array 2D).
       Toma en cuenta que el modelo en producción espera exactamente las mismas columnas que usaste para entrenar.
    2. Usa model.predict()
    3. Retorna la predicción en un diccionario, ej: {"predicted_price": 250000.0}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no se ha cargado.")

    model_input = _prepare_features_for_model(features)

    try:
        prediction = float(model.predict(model_input)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo generar la predicción: {exc}") from exc

    return {"predicted_price": prediction}

# Instrucciones para correr la API localmente:
# En la terminal, ejecuta:
# uvicorn src.api.main:app --reload
