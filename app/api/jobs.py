import json
from pathlib import Path

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from app.config import settings


router = APIRouter(
    prefix="/api/v1/tts",
    tags=["async-tts"],
)


# ============================================================
# Request
# ============================================================


class CreateTTSJobRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    voice: str = Field(
        default="default"
    )

    language: str = Field(
        default="Persian"
    )

    response_format: str = Field(
        default="mp3"
    )


# ============================================================
# Voice
# ============================================================


def load_voice(
    voice_name: str,
):

    voice_dir = (
        Path("voices") /
        voice_name
    )

    if not voice_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Voice '{voice_name}' "
                f"not found."
            ),
        )

    ref_audio = (
        voice_dir /
        "reference.mp3"
    )

    metadata_file = (
        voice_dir /
        "metadata.json"
    )

    if not ref_audio.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Reference audio not found."
            ),
        )

    if not metadata_file.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Voice metadata not found."
            ),
        )

    import json

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    return {
        "ref_audio": str(
            ref_audio.resolve()
        ),
        "ref_text": metadata.get(
            "ref_text"
        ),
        "language": metadata.get(
            "language",
            "Persian",
        ),
    }


# ============================================================
# Create Job
# ============================================================


@router.post(
    "/jobs",
    status_code=202,
)
async def create_job(
    body: CreateTTSJobRequest,
    request: Request,
):

    if len(body.text) > settings.max_input_characters:

        raise HTTPException(
            status_code=413,
            detail=(
                "Input is too long. "
                f"Maximum allowed characters: "
                f"{settings.max_input_characters}"
            ),
        )

    if body.response_format not in {
        "wav",
        "mp3",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Supported formats: "
                "wav, mp3."
            ),
        )

    voice = load_voice(
        body.voice
    )

    job_manager = (
        request.app.state.job_manager
    )

    try:

        job = await job_manager.create_job(
            text=body.text,
            voice=body.voice,
            language=body.language,
            ref_audio=voice["ref_audio"],
            ref_text=voice["ref_text"],
            response_format=body.response_format,
        )

    except RuntimeError as exc:

        if "queue is full" in str(exc).lower():

            raise HTTPException(
                status_code=429,
                detail=(
                    "TTS job queue is full."
                ),
            )

        raise

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "status_url": (
            f"/api/v1/tts/jobs/"
            f"{job.job_id}"
        ),
        "events_url": (
            f"/api/v1/tts/jobs/"
            f"{job.job_id}/events"
        ),
        "download_url": None,
    }


# ============================================================
# Job Status
# ============================================================


@router.get(
    "/jobs/{job_id}",
)
async def get_job(
    job_id: str,
    request: Request,
):

    job_manager = (
        request.app.state.job_manager
    )

    job = job_manager.get_job(
        job_id
    )

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,

        "completed_chunks":
            job.completed_chunks,

        "total_chunks":
            job.total_chunks,

        "error":
            job.error,

        "created_at":
            job.created_at,

        "started_at":
            job.started_at,

        "completed_at":
            job.completed_at,

        "download_url":
            (
                f"/api/v1/tts/jobs/"
                f"{job.job_id}/download"
            )
            if job.status == "completed"
            else None,
    }


# ============================================================
# SSE Events
# ============================================================


@router.get(
    "/jobs/{job_id}/events",
)
async def job_events(
    job_id: str,
    request: Request,
):

    job_manager = (
        request.app.state.job_manager
    )

    job = job_manager.get_job(
        job_id
    )

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    queue = await job_manager.subscribe(
        job
    )

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break

                event = await queue.get()

                yield (
                        "data: "
                        + json.dumps(event, ensure_ascii=False)
                        + "\n\n"
                )

                if event.get("status") in {"completed", "failed"}:
                    break

        finally:
            job_manager.unsubscribe(job, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# Download
# ============================================================


@router.get(
    "/jobs/{job_id}/download",
)
async def download_job(
    job_id: str,
    request: Request,
):

    job_manager = (
        request.app.state.job_manager
    )

    job = job_manager.get_job(
        job_id
    )

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if job.status != "completed":

        raise HTTPException(
            status_code=409,
            detail=(
                "Audio is not ready yet."
            ),
        )

    if not job.output_path:

        raise HTTPException(
            status_code=500,
            detail=(
                "Output file is missing."
            ),
        )

    output_path = Path(
        job.output_path
    )

    if not output_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Output file no longer exists."
            ),
        )

    if job.response_format == "mp3":

        media_type = "audio/mpeg"
        filename = "speech.mp3"

    else:

        media_type = "audio/wav"
        filename = "speech.wav"

    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=filename,
    )


# ============================================================
# Delete
# ============================================================


@router.delete(
    "/jobs/{job_id}",
)
async def delete_job(
    job_id: str,
    request: Request,
):

    job_manager = (
        request.app.state.job_manager
    )

    job = job_manager.get_job(
        job_id
    )

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if job.status in {
        "queued",
        "processing",
    }:

        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot delete an active job."
            ),
        )

    deleted = await job_manager.delete_job(
        job_id
    )

    return {
        "job_id": job_id,
        "deleted": deleted,
    }