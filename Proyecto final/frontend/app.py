import gradio as gr

from services import enviar_prediccion


def predecir_prioridad(
    asunto: str,
    contenido: str,
    canal_ticket: str,
    categoria_problema: str,
    tipo_cuenta: str,
    antiguedad_cuenta: int,
) -> str:
    """
    Función conectada a la interfaz.
    Actualmente el modelo final usa embeddings construidos desde asunto + contenido.
    Los demás campos se incluyen en la interfaz para reflejar el contexto del ticket.
    """

    prediccion = enviar_prediccion(
        asunto=asunto,
        contenido=contenido,
    )

    return f"Prioridad predicha: {prediccion}"


with gr.Blocks(
    title="ChaucherApp - Priorización de Tickets",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
    ),
) as demo:

    gr.Markdown(
        """
        # ChaucherApp - Priorización de Tickets

        Esta aplicación permite ingresar la información de un ticket de soporte
        y obtener una predicción automática del nivel de prioridad:
        **Baja, Media, Alta o Crítica**.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Atributos del ticket")

            asunto = gr.Textbox(
                label="Asunto del ticket",
                placeholder="Ej: No puedo entrar a mi cuenta",
                lines=1,
            )

            contenido = gr.Textbox(
                label="Contenido del ticket",
                placeholder=(
                    "Ej: Tengo un problema urgente porque no puedo iniciar sesión "
                    "en la app y necesito hacer una transferencia hoy."
                ),
                lines=6,
            )

            canal_ticket = gr.Dropdown(
                label="Canal del ticket",
                choices=["Whatsapp", "Correo", "Página Web"],
                value="Whatsapp",
            )

            categoria_problema = gr.Dropdown(
                label="Categoría del problema",
                choices=[
                    "Cuenta",
                    "Cobros",
                    "Fraude",
                    "Técnica",
                    "Pregunta general",
                    "Otro",
                ],
                value="Cuenta",
            )

        with gr.Column():
            gr.Markdown("## Atributos del usuario")

            tipo_cuenta = gr.Dropdown(
                label="Tipo de cuenta",
                choices=["Free", "Premium", "Business"],
                value="Free",
            )

            antiguedad_cuenta = gr.Number(
                label="Antigüedad de la cuenta en días",
                value=120,
                precision=0,
            )

            boton = gr.Button(
                "Predecir prioridad",
                variant="primary",
            )

            resultado = gr.Textbox(
                label="Resultado del modelo",
                lines=2,
                interactive=False,
            )

    gr.Markdown(
        """
        **Nota:** el modelo final seleccionado utiliza embeddings generados a partir
        del asunto y contenido del ticket. Los demás atributos se incluyen en la
        interfaz para mantener una estructura clara de ticket y usuario.
        """
    )

    boton.click(
        fn=predecir_prioridad,
        inputs=[
            asunto,
            contenido,
            canal_ticket,
            categoria_problema,
            tipo_cuenta,
            antiguedad_cuenta,
        ],
        outputs=resultado,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )