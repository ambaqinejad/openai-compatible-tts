import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class TTSJob:
    """
    A single TTS generation job.
    """

    text: str
    ref_audio: str
    language: str
    ref_text: str | None = None


class TTSWorker:

    def __init__(
        self,
        model_manager,
        max_queue_size: int = 10,
    ):
        self.model_manager = model_manager

        self.queue: asyncio.Queue[
            tuple[TTSJob, asyncio.Future]
        ] = asyncio.Queue(
            maxsize=max_queue_size
        )

        self.worker_task: asyncio.Task | None = None
        self.running = False

    async def start(self):

        if self.running:
            return

        self.running = True

        self.worker_task = asyncio.create_task(
            self._worker_loop()
        )

        print(
            "[TTS Worker] Started."
        )

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

        print(
            "[TTS Worker] Stopped."
        )

    async def submit(
        self,
        job: TTSJob,
    ) -> Any:

        if not self.running:
            raise RuntimeError(
                "TTS worker is not running."
            )

        if self.queue.full():

            raise RuntimeError(
                "TTS queue is full."
            )

        loop = asyncio.get_running_loop()

        future = loop.create_future()

        await self.queue.put(
            (job, future)
        )

        try:

            return await future

        except asyncio.CancelledError:

            if not future.done():
                future.cancel()

            raise

    async def _worker_loop(self):

        while self.running:

            job, future = await self.queue.get()

            try:

                if future.cancelled():
                    continue

                print(
                    "[TTS Worker] "
                    f"Processing job. "
                    f"Queue size: {self.queue.qsize()}"
                )

                # OmniVoice inference is synchronous,
                # so execute it in a thread instead of
                # blocking the asyncio event loop.
                audio = await asyncio.to_thread(
                    self.model_manager.generate,
                    text=job.text,
                    ref_audio=job.ref_audio,
                    language=job.language,
                    ref_text=job.ref_text,
                )

                if not future.cancelled():
                    future.set_result(audio)

            except Exception as exc:

                if not future.cancelled():
                    future.set_exception(exc)

            finally:

                self.queue.task_done()