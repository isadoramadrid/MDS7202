from pydantic import BaseModel, Field, StrictStr


class PredictionRequest(BaseModel):
    """
    Modelo de entrada para solicitar una predicción.
    Recibe el asunto y el contenido del ticket.
    """

    asunto: StrictStr = Field(
        ...,
        min_length=1,
        description="Asunto del ticket de soporte.",
    )

    contenido: StrictStr = Field(
        ...,
        min_length=1,
        description="Contenido o descripción del ticket de soporte.",
    )


class PredictionResponse(BaseModel):
    """
    Modelo de salida de la API.
    Retorna la prioridad predicha por el modelo.
    """

    prediccion: StrictStr = Field(
        ...,
        description="Nivel de prioridad predicho: Baja, Media, Alta o Critica.",
    )