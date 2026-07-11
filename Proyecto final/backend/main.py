from fastapi import FastAPI, HTTPException

from backend.generate_prediction import generate_prediction
from backend.models import PredictionRequest, PredictionResponse


app = FastAPI(
    title="API de Priorización de Tickets",
    description=(
        "API para predecir el nivel de prioridad de tickets de soporte "
        "de ChaucherApp usando un modelo entrenado de clasificación multiclase."
    ),
    version="1.0.0",
)


@app.get("/")
def home():
    """
    Endpoint simple para verificar que la API está funcionando.
    """
    return {
        "status": "ok",
        "message": "API de priorización de tickets funcionando correctamente.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Recibe un asunto y contenido de ticket, llama a generate_prediction
    y retorna la prioridad predicha.
    """

    try:
        prediccion = generate_prediction(
            asunto=request.asunto,
            contenido=request.contenido,
        )

        return PredictionResponse(prediccion=str(prediccion))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {str(error)}",
        )