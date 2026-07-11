import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai


# ---------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# El archivo .env está en la carpeta raíz del proyecto, un nivel arriba de backend
ENV_PATH = BASE_DIR.parent / ".env"

load_dotenv(ENV_PATH)


# ---------------------------------------------------------
# Constantes del modelo
# ---------------------------------------------------------

EMBEDDING_COLS = [f"embedding_dim_{i}" for i in range(1, 1025)]


# ---------------------------------------------------------
# Función para construir el texto igual al entrenamiento
# ---------------------------------------------------------

def construir_texto(asunto: str, contenido: str) -> str:
    """
    Construye el texto en el mismo formato indicado por el enunciado.
    """
    return f"Asunto_Ticket: {asunto}\nContenido_Ticket: {contenido}\n"


# ---------------------------------------------------------
# Función para generar embeddings
# ---------------------------------------------------------

def generar_embedding(texto: str) -> list[float]:
    """
    Genera un embedding de 1024 dimensiones usando gemini-embedding-001.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "No se encontró GOOGLE_API_KEY. Revisa que exista el archivo .env "
            "en la carpeta raíz del proyecto."
        )

    genai.configure(api_key=api_key)

    resultado = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texto,
        task_type="classification",
        output_dimensionality=1024,
    )

    embedding = resultado["embedding"]

    if len(embedding) != 1024:
        raise ValueError(
            f"El embedding generado tiene {len(embedding)} dimensiones, "
            "pero el modelo espera 1024."
        )

    return embedding

# ---------------------------------------------------------
# Función principal solicitada
# ---------------------------------------------------------

def generate_prediction(asunto: str, contenido: str) -> str:
    """
    Recibe asunto y contenido de un ticket, genera sus embeddings
    y retorna la prioridad predicha por el modelo entrenado.
    """

    texto = construir_texto(asunto, contenido)
    embedding = generar_embedding(texto)

    input_modelo = pd.DataFrame(
        [embedding],
        columns=EMBEDDING_COLS,
    )

    pipeline = joblib.load(MODEL_PATH)

    prediccion = pipeline.predict(input_modelo)[0]

    return prediccion


# ---------------------------------------------------------
# Ejecución de prueba cuando se corre directamente el script
# ---------------------------------------------------------

if __name__ == "__main__":
    asunto_prueba = "No puedo entrar a mi cuenta"
    contenido_prueba = (
        "Tengo un problema urgente porque no puedo iniciar sesión en la app "
        "y necesito hacer una transferencia hoy."
    )

    prediccion = generate_prediction(
        asunto=asunto_prueba,
        contenido=contenido_prueba,
    )

    print("Ticket de prueba")
    print("----------------")
    print(f"Asunto: {asunto_prueba}")
    print(f"Contenido: {contenido_prueba}")
    print(f"Predicción de prioridad: {prediccion}")