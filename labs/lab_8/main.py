import os
import pickle
from datetime import datetime

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# Creo la aplicación principal de FastAPI.
app = FastAPI(
    title="API de Potabilidad del Agua",
    description="API para predecir si una muestra de agua es potable o no usando un modelo XGBoost.",
    version="1.0.0"
)


# Defino la estructura de datos que espero recibir en el POST.
# Cada campo corresponde a una medición química del agua.
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# Defino la ruta donde está guardado mi mejor modelo.
MODEL_PATH = os.path.join("models", "best_xgboost_model.pkl")


# Defino la carpeta donde voy a guardar las predicciones.
# En Docker, esta carpeta se conectará con una carpeta local mediante un volumen.
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# Cargo el modelo una sola vez al iniciar la API.
# Así no tengo que cargarlo cada vez que alguien haga una predicción.
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.get("/")
def root():
    # Creo una ruta inicial simple para verificar que la API está funcionando.
    return {
        "mensaje": "API funcionando correctamente. Usa /home para ver la descripción o /docs para probar el modelo."
    }


@app.get("/home")
def home():
    # Entrego una descripción breve del modelo, el problema, la entrada y la salida.
    return {
        "modelo": "XGBoost Classifier optimizado con Optuna y registrado con MLflow",
        "problema": "Predecir si una muestra de agua es potable o no potable",
        "entrada": [
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity"
        ],
        "salida": {
            "potabilidad": "1 si el agua es potable, 0 si no es potable"
        }
    }


@app.post("/potabilidad/")
def predict_potability(sample: WaterSample):
    # Transformo los datos recibidos por la API a un DataFrame,
    # porque el modelo fue entrenado usando columnas con estos nombres.
    input_data = pd.DataFrame([{
        "ph": sample.ph,
        "Hardness": sample.Hardness,
        "Solids": sample.Solids,
        "Chloramines": sample.Chloramines,
        "Sulfate": sample.Sulfate,
        "Conductivity": sample.Conductivity,
        "Organic_carbon": sample.Organic_carbon,
        "Trihalomethanes": sample.Trihalomethanes,
        "Turbidity": sample.Turbidity
    }])

    # Uso el modelo cargado para predecir si el agua es potable o no.
    prediction = int(model.predict(input_data)[0])

    # Creo un registro con los datos recibidos, la predicción y la fecha.
    # Esto me permite demostrar que el contenedor puede persistir resultados.
    record = input_data.copy()
    record["potabilidad"] = prediction
    record["fecha_prediccion"] = datetime.now().isoformat()

    # Defino el archivo donde guardaré todas las predicciones.
    output_path = os.path.join(DATA_DIR, "predicciones.csv")

    # Guardo la predicción en modo append.
    # Si el archivo no existe, escribo el encabezado; si ya existe, solo agrego una fila.
    record.to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        index=False
    )

    # Retorno la predicción como entero para que la respuesta sea clara.
    return {
        "potabilidad": prediction
    }


# Con este bloque puedo levantar el servidor ejecutando directamente:
# python main.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)