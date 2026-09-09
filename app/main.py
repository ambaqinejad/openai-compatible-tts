from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.core.model import OmniVoiceModelManager
from app.core.worker import TTSWorker
from app.api.speech import router as speech_router


model_manager = OmniVoiceModelManager()

tts_worker = TTSWorker(
    model_manager=model_manager,
    max_queue_size=settings.max_queue_size,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting OmniVoice API")
    print("=" * 60)

    try:

        model_manager.load()

        app.state.model_manager = model_manager

        await tts_worker.start()

        app.state.tts_worker = tts_worker

        print("=" * 60)
        print("OmniVoice API is READY")
        print("=" * 60)

    except Exception as exc:

        print("=" * 60)
        print("FAILED TO START OMNIVOICE API")
        print(exc)
        print("=" * 60)

        raise

    yield

    print(
        "Shutting down OmniVoice API..."
    )

    await tts_worker.stop()

    model_manager.unload()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


app.include_router(
    speech_router
)


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/ready")
async def ready():

    status = model_manager.status()

    if not status["loaded"]:

        return {
            "status": "not_ready",
            **status,
        }

    return {
        "status": "ready",
        "worker_running": tts_worker.running,
        "queue_size": tts_worker.queue.qsize(),
        **status,
    }