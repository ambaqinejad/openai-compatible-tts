from pathlib import Path
import json
import uuid

import soundfile as sf

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings

from app.audio.chunker import chunk_text
from app.audio.processor import concatenate_audio
from app.audio.ffmpeg import convert_wav_to_mp3

router = APIRouter(
    prefix="/v1/audio",
    tags=["audio"],
)


VOICE_ROOT = Path("voices")
OUTPUT_ROOT = Path("output")


class SpeechRequest(BaseModel):

    model: str = Field(
        default="omnivoice",
        description="TTS model name",
    )

    input: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize",
    )

    voice: str = Field(
        default="default",
        description="Voice name",
    )

    response_format: str = Field(
        default="wav",
        description="Audio format",
    )

    language: str = Field(
        default="Persian",
        description="Input language",
    )


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
                "Supported response formats: wav, mp3."
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

    wav_path = (
        output_dir /
        f"{request_id}.wav"
    )

    mp3_path = (
        output_dir /
        f"{request_id}.mp3"
    )

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

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"[TTS] request_id={request_id} "
                f"chunk={index}/{len(chunks)} "
                f"characters={len(chunk)}"
            )

            audio = model_manager.generate(
                text=chunk,
                ref_audio=voice["ref_audio"],
                language=body.language,
                ref_text=voice["ref_text"],
            )

            if not audio:

                raise RuntimeError(
                    f"OmniVoice returned empty audio "
                    f"for chunk {index}."
                )

            audio_chunks.append(
                audio[0]
            )

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
                    "X-Request-ID": request_id,
                },
            )

        convert_wav_to_mp3(
            input_path=wav_path,
            output_path=mp3_path,
            bitrate=settings.mp3_bitrate,
        )

        wav_path.unlink()

        return FileResponse(
            path=str(mp3_path),
            media_type="audio/mpeg",
            filename="speech.mp3",
            headers={
                "X-Request-ID": request_id,
            },
        )

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
            detail=f"TTS generation failed: {exc}",
        )