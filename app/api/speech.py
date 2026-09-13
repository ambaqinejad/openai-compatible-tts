from pathlib import Path
import json
import uuid

import soundfile as sf

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.core.worker import TTSJob

from app.audio.chunker import chunk_text
from app.audio.processor import concatenate_audio
from app.audio.ffmpeg import convert_wav_to_mp3


router = APIRouter(
    prefix="/v1/audio",
    tags=["audio"],
)


VOICE_ROOT = Path("voices")
OUTPUT_ROOT = Path("output")
CLONE_ROOT = Path("output/voice-clone")


# ============================================================
# Existing OpenWebUI TTS API
# ============================================================

class SpeechRequest(BaseModel):
    model: str = Field(default="omnivoice")
    input: str = Field(..., min_length=1)
    voice: str = Field(default="default")
    response_format: str = Field(default="wav")
    language: str = Field(default="Persian")


def load_voice(voice_name: str):
    voice_dir = VOICE_ROOT / voice_name

    if not voice_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found.",
        )

    ref_audio = voice_dir / "reference.mp3"
    metadata_file = voice_dir / "metadata.json"

    if not ref_audio.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Reference audio not found "
                f"for voice '{voice_name}'."
            ),
        )

    if not metadata_file.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Voice metadata not found "
                f"for voice '{voice_name}'."
            ),
        )

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    return {
        "ref_audio": str(ref_audio.resolve()),
        "ref_text": metadata.get("ref_text"),
        "language": metadata.get(
            "language",
            "Persian",
        ),
    }


@router.post("/speech")
async def create_speech(
    request: Request,
    body: SpeechRequest,
):
    if len(body.input) > settings.max_input_characters:
        raise HTTPException(
            status_code=413,
            detail=(
                "Input is too long. "
                f"Maximum allowed characters: "
                f"{settings.max_input_characters}"
            ),
        )

    if body.model != "omnivoice":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {body.model}",
        )

    if body.response_format not in {
        "wav",
        "mp3",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported response formats: "
                "wav, mp3."
            ),
        )

    voice = load_voice(body.voice)

    model_manager = request.app.state.model_manager

    if not model_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail="TTS model is not ready.",
        )

    request_id = uuid.uuid4().hex

    output_dir = OUTPUT_ROOT
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wav_path = output_dir / f"{request_id}.wav"
    mp3_path = output_dir / f"{request_id}.mp3"

    try:
        chunks = chunk_text(
            body.input,
            max_length=settings.max_chunk_characters,
        )

        if not chunks:
            raise ValueError(
                "Input text produced no chunks."
            )

        print(
            f"[TTS] request_id={request_id} "
            f"characters={len(body.input)} "
            f"chunks={len(chunks)} "
            f"voice={body.voice} "
            f"format={body.response_format}"
        )

        audio_chunks = []

        tts_worker = request.app.state.tts_worker

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            print(
                f"[TTS] request_id={request_id} "
                f"chunk={index}/{len(chunks)} "
                f"characters={len(chunk)} "
                f"queue={tts_worker.queue.qsize()}"
            )

            job = TTSJob(
                text=chunk,
                ref_audio=voice["ref_audio"],
                language=body.language,
                ref_text=voice["ref_text"],
            )

            try:
                audio = await tts_worker.submit(
                    job
                )

            except RuntimeError as exc:
                if "queue is full" in str(exc).lower():
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "TTS queue is full. "
                            "Please try again later."
                        ),
                    )

                raise

            if not audio:
                raise RuntimeError(
                    f"OmniVoice returned empty audio "
                    f"for chunk {index}."
                )

            audio_chunks.append(audio[0])

        final_audio = concatenate_audio(
            audio_chunks,
            silence_ms=80,
            sample_rate=settings.output_sample_rate,
        )

        sf.write(
            str(wav_path),
            final_audio,
            settings.output_sample_rate,
            format="WAV",
        )

        if body.response_format == "wav":
            return FileResponse(
                path=str(wav_path),
                media_type="audio/wav",
                filename="speech.wav",
                headers={
                    "X-Request-ID": request_id
                },
            )

        convert_wav_to_mp3(
            input_path=wav_path,
            output_path=mp3_path,
            bitrate=settings.mp3_bitrate,
        )

        wav_path.unlink(
            missing_ok=True
        )

        return FileResponse(
            path=str(mp3_path),
            media_type="audio/mpeg",
            filename="speech.mp3",
            headers={
                "X-Request-ID": request_id
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"[TTS] failed "
            f"request_id={request_id} "
            f"error={exc}"
        )

        if wav_path.exists():
            wav_path.unlink()

        if mp3_path.exists():
            mp3_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                f"TTS generation failed: {exc}"
            ),
        )


# ============================================================
# Zero-Shot Voice Cloning API
# ============================================================

@router.post("/voice-clone")
async def create_voice_clone(
    request: Request,

    text: str = Form(...),
    reference_text: str = Form(...),
    language: str = Form(default="English"),
    response_format: str = Form(default="wav"),

    reference_audio: UploadFile = File(...),
):
    """
    Generate speech using zero-shot voice cloning.

    The uploaded reference audio is used as the speaker
    reference and reference_text must be the transcription
    of that audio.
    """

    # --------------------------------------------------------
    # Validate text
    # --------------------------------------------------------

    text = text.strip()
    reference_text = reference_text.strip()
    language = language.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    if not reference_text:
        raise HTTPException(
            status_code=400,
            detail="Reference text cannot be empty.",
        )

    if len(text) > settings.max_input_characters:
        raise HTTPException(
            status_code=413,
            detail=(
                "Input text is too long. "
                f"Maximum allowed characters: "
                f"{settings.max_input_characters}"
            ),
        )

    # --------------------------------------------------------
    # Validate response format
    # --------------------------------------------------------

    if response_format not in {
        "wav",
        "mp3",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported response formats: "
                "wav, mp3."
            ),
        )

    # --------------------------------------------------------
    # Validate reference audio
    # --------------------------------------------------------

    if not reference_audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Reference audio filename is missing.",
        )

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    }

    extension = Path(
        reference_audio.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported reference audio format. "
                f"Supported formats: "
                f"{', '.join(sorted(allowed_extensions))}"
            ),
        )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    model_manager = request.app.state.model_manager

    if not model_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail="TTS model is not ready.",
        )

    # --------------------------------------------------------
    # Request ID / paths
    # --------------------------------------------------------

    request_id = uuid.uuid4().hex

    request_dir = (
        CLONE_ROOT / request_id
    )

    request_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_path = (
        request_dir
        / f"reference{extension}"
    )

    wav_path = (
        request_dir
        / f"{request_id}.wav"
    )

    mp3_path = (
        request_dir
        / f"{request_id}.mp3"
    )

    try:

        # ----------------------------------------------------
        # Save uploaded reference audio
        # ----------------------------------------------------

        with open(
            reference_path,
            "wb",
        ) as f:

            while True:
                chunk = await reference_audio.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                f.write(chunk)

        await reference_audio.close()

        print(
            f"[VOICE CLONE] "
            f"request_id={request_id} "
            f"text_characters={len(text)} "
            f"reference={reference_path} "
            f"language={language} "
            f"format={response_format}"
        )

        # ----------------------------------------------------
        # Split long text
        # ----------------------------------------------------

        chunks = chunk_text(
            text,
            max_length=settings.max_chunk_characters,
        )

        if not chunks:
            raise ValueError(
                "Input text produced no chunks."
            )

        print(
            f"[VOICE CLONE] "
            f"request_id={request_id} "
            f"chunks={len(chunks)}"
        )

        # ----------------------------------------------------
        # Generate audio
        # ----------------------------------------------------

        audio_chunks = []

        tts_worker = request.app.state.tts_worker

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"[VOICE CLONE] "
                f"request_id={request_id} "
                f"chunk={index}/{len(chunks)}"
            )

            job = TTSJob(
                text=chunk,
                ref_audio=str(
                    reference_path.resolve()
                ),
                language=language,
                ref_text=reference_text,
            )

            try:
                audio = await tts_worker.submit(
                    job
                )

            except RuntimeError as exc:

                if "queue is full" in str(exc).lower():
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "TTS queue is full. "
                            "Please try again later."
                        ),
                    )

                raise

            if not audio:
                raise RuntimeError(
                    f"OmniVoice returned empty audio "
                    f"for chunk {index}."
                )

            audio_chunks.append(
                audio[0]
            )

        # ----------------------------------------------------
        # Concatenate generated chunks
        # ----------------------------------------------------

        final_audio = concatenate_audio(
            audio_chunks,
            silence_ms=80,
            sample_rate=settings.output_sample_rate,
        )

        # ----------------------------------------------------
        # Save WAV
        # ----------------------------------------------------

        sf.write(
            str(wav_path),
            final_audio,
            settings.output_sample_rate,
            format="WAV",
        )

        # ----------------------------------------------------
        # Return WAV
        # ----------------------------------------------------

        if response_format == "wav":

            return FileResponse(
                path=str(wav_path),
                media_type="audio/wav",
                filename="voice-clone.wav",
                headers={
                    "X-Request-ID": request_id
                },
            )

        # ----------------------------------------------------
        # Convert WAV -> MP3
        # ----------------------------------------------------

        convert_wav_to_mp3(
            input_path=wav_path,
            output_path=mp3_path,
            bitrate=settings.mp3_bitrate,
        )

        wav_path.unlink(
            missing_ok=True
        )

        return FileResponse(
            path=str(mp3_path),
            media_type="audio/mpeg",
            filename="voice-clone.mp3",
            headers={
                "X-Request-ID": request_id
            },
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"[VOICE CLONE] failed "
            f"request_id={request_id} "
            f"error={exc}"
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        for path in (
            reference_path,
            wav_path,
            mp3_path,
        ):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        try:
            request_dir.rmdir()
        except OSError:
            pass

        raise HTTPException(
            status_code=500,
            detail=(
                f"Voice cloning failed: {exc}"
            ),
        )