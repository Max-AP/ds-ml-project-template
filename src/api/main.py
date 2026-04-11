"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Inicializamos la app
app = FastAPI(title="API de Predicción de Precios de Vivienda (California)", version="1.0")

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
    # Añade cualquier variable categórica o enriquecida que el modelo requiera
    # ej: ocean_proximity: str 
    # ej: rooms_per_household: float

# Variable global para cargar el modelo
# IMPORTANTE: Asegúrate de guardar tu modelo en "models/best_model.pkl" o ajusta la ruta
model = None

@app.on_event("startup")
def load_model():
    """
    Carga el modelo globalmente al iniciar el servidor usando joblib.
    """
    global model
    try:
        model = joblib.load("models/best_model.pkl")
    except Exception as e:
        print("Advertencia: No se pudo cargar el modelo. ¿Ya lo entrenaste y guardaste?")

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
        return {"error": "El modelo no se ha cargado."}
    
    # Tu código aquí para predecir
    prediction = 0.0 # Reemplazar con model.predict()
    
    return {"predicted_price": prediction}

# Instrucciones para correr la API localmente:
# En la terminal, ejecuta:
# uvicorn src.api.main:app --reload
