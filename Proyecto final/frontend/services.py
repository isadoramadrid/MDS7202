import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Cargar variables de entorno desde el .env ubicado en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"

load_dotenv(ENV_PATH)


BACKEND_URI = os.getenv("BACKEND_URI", "http://127.0.0.1:8000")


def enviar_prediccion(asunto: str, contenido: str) -> str:
    """
    Envía el asunto y contenido del ticket al endpoint /predict
    del backend y retorna la predicción del modelo.
    """

    if not asunto or not asunto.strip():
        return "Error: debes ingresar un asunto para el ticket."

    if not contenido or not contenido.strip():
        return "Error: debes ingresar el contenido del ticket."

    payload = {
        "asunto": asunto,
        "contenido": contenido,
    }

    try:
        response = requests.post(
            f"{BACKEND_URI}/predict",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()
        prediccion = data.get("prediccion", "Sin predicción")

        return prediccion

    except requests.exceptions.RequestException as error:
        return f"Error al conectar con la API: {error}"

    except Exception as error:
        return f"Error inesperado: {error}"