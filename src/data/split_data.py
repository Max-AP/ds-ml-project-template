"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con partición de entrenamiento/prueba.
    3. Se usa `StratifiedShuffleSplit` basado en la variable del ingreso medio (`median_income`)
       para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    housing = pd.read_csv(raw_data_path)

    # Split estratificado por nivel de ingreso para mejorar representatividad
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(housing, housing["income_cat"]))

    train_set = housing.iloc[train_idx].drop(columns=["income_cat"])
    test_set = housing.iloc[test_idx].drop(columns=["income_cat"])

    Path(interim_data_path).mkdir(parents=True, exist_ok=True)
    train_set.to_csv(Path(interim_data_path) / "train_set.csv", index=False)
    test_set.to_csv(Path(interim_data_path) / "test_set.csv", index=False)


if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
    print("Datos divididos y guardados en data/interim/")
