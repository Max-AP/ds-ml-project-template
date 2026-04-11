"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Maneja los valores faltantes.
       Puedes llenarlos con la mediana de la columna.
    2. Retorna el DataFrame limpio.
    """
    # Tu código aquí
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Agrega nuevas variables derivando las existentes, por ejemplo:
       - 'rooms_per_household' = total_rooms / households
       - 'population_per_household' = population / households
       - 'bedrooms_per_room' = total_bedrooms / total_rooms
    2. Retorna el DataFrame enriquecido.
    """
    # Tu código aquí
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que toma el DataFrame crudo y aplica limpieza y enriquecimiento.
    """
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)
    
    # IMPORTANTE: Aquí los alumnos deberían añadir codificación de variables categóricas
    # (ej. get_dummies para 'ocean_proximity') si no usan Pipelines de Scikit-Learn.
    
    return df_featured

if __name__ == "__main__":
    print("Módulo de feature engineering... (Falta el código!)")
