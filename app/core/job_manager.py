import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

from app.audio.chunker import chunk_text
from app.audio.processor import concatenate_audio
from app.audio.ffmpeg import convert_wav_to_mp3
from app.config import settings


OUTPUT_ROOT = Path("output/jobs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TTSJobState:
    job_id: str
    text: str
    voice: str
    language: str
    ref_audio: str
    ref_text: str | None

    response_format: str = "mp3"

    status: str = "queued"
    progress: int = 0

    total_chunks: int = 0
    completed_chunks: int = 0

    output_path: str | None = None
    error: str | None = None

    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None

    subscribers: set[asyncio.Queue] = field(default_factory=set)


class TTSJobManager:

    def __init__(
        self,
        model_manager,
        max_queue_size: int = 10,
    ):
        self.model_manager = model_manager

        self.jobs: dict[str, TTSJobState] = {}

        self.queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=max_queue_size
        )

        self.worker_task: asyncio.Task | None = None
        self.running = False

    # ==========================================================
    # Lifecycle
    # ==========================================================

    async def start(self):
        if self.running:
            return

        self.running = True

        self.worker_task = asyncio.create_task(
            self._worker_loop()
        )

        print("[Job Manager] Started.")

    async def stop(self):
        if not self.running:
            return

        self.running = False

        if self.worker_task:
            self.worker_task.cancel()

            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

            self.worker_task = None

        print("[Job Manager] Stopped.")

    # ==========================================================
    # Create
    # ==========================================================

    async def create_job(
        self,
        text: str,
        voice: str,
        language: str,
        ref_audio: str,
        ref_text: str | None,
        response_format: str,
    ) -> TTSJobState:

        if self.queue.full():
            raise RuntimeError("TTS job queue is full.")

        job_id = uuid.uuid4().hex

        job = TTSJobState(
            job_id=job_id,
            text=text,
            voice=voice,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            response_format=response_format,
        )

        self.jobs[job_id] = job

        await self.queue.put(job_id)

        await self._publish(
            job,
            {
                "type": "status",
                "status": "queued",
                "progress": 0,
                "job_id": job_id,
            },
        )

        print(
            f"[Job Manager] Job created "
            f"id={job_id} "
            f"characters={len(text)}"
        )

        return job

    # ==========================================================
    # Worker
    # ==========================================================

    async def _worker_loop(self):

        while self.running:

            job_id = await self.queue.get()

            try:
                job = self.jobs.get(job_id)

                if job is None:
                    continue

                await self._process_job(job)

            except Exception as exc:
                print(
                    f"[Job Manager] Worker error: {exc}"
                )

            finally:
                self.queue.task_done()

    # ==========================================================
    # Process
    # ==========================================================

    async def _process_job(
        self,
        job: TTSJobState,
    ):

        job.status = "processing"
        job.started_at = utc_now()

        await self._publish(
            job,
            {
                "type": "status",
                "status": "processing",
                "progress": 0,
                "job_id": job.job_id,
            },
        )

        try:

            chunks = chunk_text(
                job.text,
                max_length=settings.max_chunk_characters,
            )

            if not chunks:
                raise ValueError(
                    "Input text produced no chunks."
                )

            job.total_chunks = len(chunks)

            await self._publish(
                job,
                {
                    "type": "chunks",
                    "job_id": job.job_id,
                    "total_chunks": job.total_chunks,
                },
            )

            audio_chunks = []

            for index, chunk in enumerate(
                chunks,
                start=1,
            ):

                print(
                    f"[TTS Job] "
                    f"id={job.job_id} "
                    f"chunk={index}/{len(chunks)}"
                )

                audio = await asyncio.to_thread(
                    self.model_manager.generate,
                    text=chunk,
                    ref_audio=job.ref_audio,
                    language=job.language,
                    ref_text=job.ref_text,
                )

                if not audio:
                    raise RuntimeError(
                        f"OmniVoice returned empty audio "
                        f"for chunk {index}."
                    )

                audio_chunks.append(audio[0])

                job.completed_chunks = index

                progress = int(
                    index / job.total_chunks * 100
                )

                job.progress = progress

                await self._publish(
                    job,
                    {
                        "type": "progress",
                        "job_id": job.job_id,
                        "status": "processing",
                        "progress": progress,
                        "completed_chunks": index,
                        "total_chunks": job.total_chunks,
                    },
                )

            # ==================================================
            # Concatenate
            # ==================================================

            await self._publish(
                job,
                {
                    "type": "status",
                    "job_id": job.job_id,
                    "status": "processing",
                    "progress": 98,
                    "message": "Combining audio chunks.",
                },
            )

            final_audio = concatenate_audio(
                audio_chunks,
                silence_ms=80,
                sample_rate=settings.output_sample_rate,
            )

            output_dir = (
                OUTPUT_ROOT / job.job_id
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            wav_path = (
                output_dir /
                f"{job.job_id}.wav"
            )

            mp3_path = (
                output_dir /
                f"{job.job_id}.mp3"
            )

            # ==================================================
            # WAV
            # ==================================================

            sf.write(
                str(wav_path),
                final_audio,
                settings.output_sample_rate,
                format="WAV",
            )

            # ==================================================
            # MP3
            # ==================================================

            if job.response_format == "mp3":

                convert_wav_to_mp3(
                    input_path=wav_path,
                    output_path=mp3_path,
                    bitrate=settings.mp3_bitrate,
                )

                wav_path.unlink(
                    missing_ok=True
                )

                job.output_path = str(
                    mp3_path
                )

            else:

                job.output_path = str(
                    wav_path
                )

            # ==================================================
            # Completed
            # ==================================================

            job.progress = 100
            job.status = "completed"
            job.completed_at = utc_now()

            await self._publish(
                job,
                {
                    "type": "completed",
                    "job_id": job.job_id,
                    "status": "completed",
                    "progress": 100,
                    "download_url": (
                        f"/api/v1/tts/jobs/"
                        f"{job.job_id}/download"
                    ),
                },
            )

            print(
                f"[TTS Job] Completed "
                f"id={job.job_id}"
            )

        except Exception as exc:

            job.status = "failed"
            job.error = str(exc)
            job.completed_at = utc_now()

            await self._publish(
                job,
                {
                    "type": "failed",
                    "job_id": job.job_id,
                    "status": "failed",
                    "error": str(exc),
                },
            )

            print(
                f"[TTS Job] Failed "
                f"id={job.job_id} "
                f"error={exc}"
            )

    # ==========================================================
    # Events
    # ==========================================================

    async def subscribe(
            self,
            job: TTSJobState,
    ) -> asyncio.Queue:

        queue = asyncio.Queue()

        job.subscribers.add(queue)

        event = {
            "type": "current_state",
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "completed_chunks": job.completed_chunks,
            "total_chunks": job.total_chunks,
            "error": job.error,
            "download_url": (
                f"/api/v1/tts/jobs/{job.job_id}/download"
                if job.status == "completed"
                else None
            ),
        }

        await queue.put(event)

        return queue

    def unsubscribe(
        self,
        job: TTSJobState,
        queue: asyncio.Queue,
    ):

        job.subscribers.discard(queue)

    async def _publish(
        self,
        job: TTSJobState,
        event: dict[str, Any],
    ):

        for queue in list(job.subscribers):

            try:
                queue.put_nowait(event)

            except asyncio.QueueFull:
                pass

    # ==========================================================
    # Get
    # ==========================================================

    def get_job(
        self,
        job_id: str,
    ) -> TTSJobState | None:

        return self.jobs.get(job_id)

    # ==========================================================
    # Delete
    # ==========================================================

    async def delete_job(
        self,
        job_id: str,
    ) -> bool:

        job = self.jobs.get(job_id)

        if job is None:
            return False

        if job.status in {
            "queued",
            "processing",
        }:
            return False

        if job.output_path:

            path = Path(
                job.output_path
            )

            if path.exists():
                path.unlink()

            try:
                path.parent.rmdir()
            except OSError:
                pass

        del self.jobs[job_id]

        return True