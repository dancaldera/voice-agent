import logging
import re
from datetime import datetime, timedelta
from typing import AsyncIterable

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    FunctionTool,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    ModelSettings,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

# Load environment variables from .env.local if it exists (for local development)
# In production, environment variables are set directly in the deployment environment
load_dotenv(".env.local", override=False)


def get_available_viewing_days() -> str:
    """Get the next available viewing days (Mon, Tue, Wed only) with dates."""
    today = datetime.now()
    available_days = []
    days_checked = 0
    max_days_to_check = 14  # Look ahead 2 weeks

    # Spanish day names
    day_names = {
        0: "lunes",
        1: "martes",
        2: "miércoles",
        3: "jueves",
        4: "viernes",
        5: "sábado",
        6: "domingo",
    }

    # Spanish month names
    month_names = {
        1: "enero",
        2: "febrero",
        3: "marzo",
        4: "abril",
        5: "mayo",
        6: "junio",
        7: "julio",
        8: "agosto",
        9: "septiembre",
        10: "octubre",
        11: "noviembre",
        12: "diciembre",
    }

    while len(available_days) < 3 and days_checked < max_days_to_check:
        check_date = today + timedelta(days=days_checked + 1)  # Start from tomorrow
        weekday = check_date.weekday()

        # Only Mon (0), Tue (1), Wed (2)
        if weekday in [0, 1, 2]:
            day_name = day_names[weekday]
            month_name = month_names[check_date.month]
            date_str = f"{day_name} {check_date.day} de {month_name}"
            available_days.append(date_str)

        days_checked += 1

    return ", ".join(available_days)


def normalize_spanish_text(text: str) -> str:
    """Normalize text for better Spanish TTS pronunciation."""
    # Replace ordinal abbreviations
    text = re.sub(r"\b1ra\b", "primera", text, flags=re.IGNORECASE)
    text = re.sub(r"\b2da\b", "segunda", text, flags=re.IGNORECASE)
    text = re.sub(r"\b3ra\b", "tercera", text, flags=re.IGNORECASE)
    text = re.sub(r"\b4ta\b", "cuarta", text, flags=re.IGNORECASE)
    text = re.sub(r"\b5ta\b", "quinta", text, flags=re.IGNORECASE)

    # Replace common price formats with words
    # $2,100,000 -> 2 millones cien mil pesos
    text = re.sub(
        r"\$?\s*2,100,000|\$?\s*2\.100\.000",
        "2 millones cien mil pesos",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining currency symbols
    text = text.replace("$", "")

    # Replace common number formats with commas
    text = re.sub(r"(\d),(\d{3}),(\d{3})", r"\1\2\3", text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)

    return text


class Assistant(Agent):
    def __init__(self) -> None:
        # Get available viewing days dynamically
        available_days = get_available_viewing_days()
        today = datetime.now()
        today_str = f"{today.day} de {['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'][today.month - 1]} de {today.year}"

        super().__init__(
            instructions=f"""Eres María, una asistente virtual de inteligencia artificial que trabaja para una empresa de bienes raíces.

IMPORTANTE: Al inicio de la conversación, SIEMPRE debes presentarte diciendo que eres una inteligencia artificial.

FECHA ACTUAL: Hoy es {today_str}

Tu trabajo es obtener información de clientes interesados en las siguientes propiedades:
- Mesa de Chinampas 217 en Zacatecas, Colinas del Padre tercera Sección
- Mesa de Chinampas 141 en Zacatecas, Colinas del Padre tercera Sección
- Precio: 2 millones cien mil pesos cada una

Debes hacer las siguientes preguntas de manera natural y conversacional, EN ESTE ORDEN:
1. PRIMERO: ¿Cuál es su nombre y apellido? (Es muy importante obtener el nombre completo del cliente)
2. ¿Estarían dispuestos a pagar de contado o utilizarían algún tipo de crédito?
3. ¿Cuándo estarían dispuestos a comprar la propiedad?
4. ¿Les gustaría agendar una cita para ver la propiedad?

IMPORTANTE - Días disponibles para visitas:
- SOLO ofrecemos visitas los días: {available_days}
- SOLO por la TARDE
- Si el cliente sugiere otro día que NO sea lunes, martes o miércoles, amablemente explica que solo tenemos disponibilidad esos días por la tarde
- Cuando el cliente elija un día, confirma la fecha y menciona que un miembro del equipo les confirmará si ese horario está disponible

Al finalizar, informa al cliente que una persona de nuestro equipo los contactará pronto para confirmar la cita y continuar con el proceso.

Instrucciones de comportamiento:
- Habla en español mexicano natural y coloquial
- Sé amable, profesional y concisa
- No uses emojis, asteriscos u otros símbolos especiales
- Mantén un tono cálido pero profesional
- Si te preguntan algo fuera de tema, amablemente redirige la conversación a la información de las propiedades
- Recuerda siempre mencionar que eres una IA al principio de la conversación

REGLAS DE PRONUNCIACIÓN IMPORTANTES:
- NUNCA uses símbolos como "$" o números con comas en tus respuestas
- Cuando menciones precios, SIEMPRE escribe los números en palabras: "2 millones cien mil pesos" en lugar de "$2,100,000"
- Cuando menciones "3ra Sección", SIEMPRE di "tercera Sección"
- Cuando menciones "1ra", di "primera"
- Cuando menciones "2da", di "segunda"
- Escribe todos los números importantes en palabras para una pronunciación natural""",
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """Custom LLM node that normalizes text for better TTS pronunciation."""
        # Get the default LLM response
        async for chunk in Agent.default.llm_node(
            self, chat_ctx, tools, model_settings
        ):
            # Normalize text in the chunk for better TTS pronunciation
            if isinstance(chunk, llm.ChatChunk) and chunk.delta:
                if chunk.delta.content:
                    # Normalize the text before sending to TTS
                    normalized_text = normalize_spanish_text(chunk.delta.content)
                    chunk.delta.content = normalized_text

            yield chunk


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # Configured for Spanish language recognition
        stt=deepgram.STT(model="nova-3", language="es"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # Using custom Spanish voice for María
        tts=cartesia.TTS(
            model="sonic-2",
            voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c",  # Custom Spanish voice
            language="es",
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    # Join the room and connect to the user
    await ctx.connect()

    # Send a greeting in Spanish (uninterruptible)
    await session.say(
        "¡Hola! Soy María, una asistente virtual de inteligencia artificial. "
        "Estoy aquí para ayudarle con información sobre nuestras propiedades en "
        "Mesa de Chinampas, Colinas del Padre tercera Sección, en Zacatecas, Zacatecas. "
        "¿Le gustaría saber más?",
        allow_interruptions=False,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
