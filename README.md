# Proyecto Final de Fundamentos de Ciencia de Datos - USFQ

Este es tu entorno de trabajo (*boilerplate*) profesional. Ustedes podrán utilizar este repositorio como base para estructurar su trabajo final o cualquier producto de datos a nivel productivo.

El objetivo de este proyecto es predecir los precios medios de las viviendas en distritos de California, partiendo desde los datos crudos hasta un despliegue básico en una API.

## Estructura del Repositorio

- `data/`: Contiene los datos crudos, intermedios y procesados (NO se suben a Git).
- `notebooks/`: Espacio interactivo en Jupyter para EDA y experimentación de modelos.
- `src/`: Código fuente modular en Python (carga, procesamiento, entrenamiento y API).
- `models/`: Donde debes guardar tu modelo predictivo entrenado (ej. `.pkl`).

---

## Instrucciones: Proyecto Final Fundamentos de DS

Tu objetivo es completar el código faltante en los `notebooks/` y `src/` guiándote por las instrucciones (`docstrings`) dejadas en cada archivo.

### Fase 1: Recolección y Análisis Exploratorio
1. Ejecuta y completa **`src/data/make_dataset.py`** para descargar los datos (`housing.tgz`) en la carpeta `data/raw/`.
2. Completa **`src/data/split_data.py`** para realizar la partición de datos (Train/Test) asegurando no tener fuga de datos (*data leakage*). Guarda los resultados en `data/interim/`.
3. Dirígete a **`notebooks/01_exploracion.ipynb`**. Realiza el Análisis Exploratorio (EDA) a profundidad sobre tus datos de entrenamiento. Visualiza problemas de calidad y documenta tus hallazgos.

### Fase 2: Ingeniería de Variables y Limpieza
4. Experimenta en **`notebooks/02_limpieza_enriquecimiento.ipynb`**. Resuelve el manejo de faltantes, codificación de textos categóricos y crea nuevas variables combinadas (ej. recámaras por hogar).
5. Traslada la lógica funcional que aprendiste al script **`src/features/build_features.py`** para que este proceso se pueda repetir fácilmente con datos nuevos.

### Fase 3: Modelado y Optimización
6. Experimenta en **`notebooks/03_experimentacion.ipynb`**. Entrena un `LinearRegression`, `SGDRegressor`, `DecisionTree` y `RandomForest`. Haz un *fine-tuning* (grid search) al ganador, justifica tu elección y evalúa el subajuste/sobreajuste contra tu set de prueba.
7. Modifica **`src/models/train_model.py`** para configurar el flujo final: entrenar tu modelo ganador sobre las variables procesadas y guardarlo (serializarlo con `joblib`) en la carpeta `models/`.

### Fase 4: Despliegue en Producción
8. Abre **`src/api/main.py`**. En este script base de FastAPI, carga tu modelo y procesa las peticiones (POST) para regresar una predicción de precio en tiempo real.

---

## Instrucciones de Configuración Inicial

1. **Clonar este repositorio** en tu máquina local.
2. **Crear un entorno virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Mac/Linux
   # .venv\Scripts\activate   # En Windows
   ```
3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Rúbrica de Calificación (100 Puntos Totales)

Este proyecto busca evaluar que puedas entablar un puente entre la teoría de la Maestría y la ingeniería de datos en la industria.

| Criterio | Descripción | Puntos |
| :--- | :--- | :---: |
| **1. Análisis Exploratorio y Calidad de Datos** | EDA profundo, uso de visualizaciones (distribución, mapas geográficos, dispersión), hallazgos de métricas y validación correcta de Train/Test Split. | **15 pts** |
| **2. Limpieza y Feature Engineering** | Correcto manejo de imputaciones, codificación nominal/ordinal justificada, y creación de nuevas variables. Integración correcta en los scripts base (`src/features`). | **15 pts** |
| **3. Experimentación y Selección de Modelos** | Prueba exhaustiva de modelos (SGD, Árboles, Regresión Lineal, Ensembles). Adecuada selección de hiperparámetros de validación cruzada. Análisis riguroso del Benchmark final. | **25 pts** |
| **4. Despliegue Básico con FastAPI** | El modelo debe cargarse sin errores en la API, y recibir parámetros (`JSON`) respondiendo predicciones congruentes por medio del protocolo HTTP. | **15 pts** |
| **5. Presentación del Proyecto** | Comunicación clara, entendimiento del caso de negocio frente a stakeholders. La manera cómo se transmiten los resultados, seguridad, elocuencia y conclusiones ejecutivas del equipo. | **30 pts** |

¡Mucho éxito!
