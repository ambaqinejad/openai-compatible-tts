import gc
import os
from pathlib import Path

import torch

from app.config import settings


class OmniVoiceModelManager:

    DTYPE_MAP = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    def __init__(self):
        self.model = None
        self.loaded = False

        self.device = settings.device
        self.dtype = self._get_dtype(settings.dtype)

    def _get_dtype(self, dtype_name: str):

        dtype_name = dtype_name.lower().strip()

        if dtype_name not in self.DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype_name}. "
                f"Supported: {list(self.DTYPE_MAP.keys())}"
            )

        return self.DTYPE_MAP[dtype_name]

    def _validate_paths(self):

        omnivoice_path = Path(settings.omnivoice_model_path)
        whisper_path = Path(settings.whisper_model_path)

        if not omnivoice_path.exists():
            raise FileNotFoundError(
                f"OmniVoice model path does not exist: {omnivoice_path}"
            )

        if not omnivoice_path.is_dir():
            raise NotADirectoryError(
                f"OmniVoice model path is not a directory: {omnivoice_path}"
            )

        if not whisper_path.exists():
            raise FileNotFoundError(
                f"Whisper model path does not exist: {whisper_path}"
            )

        if not whisper_path.is_dir():
            raise NotADirectoryError(
                f"Whisper model path is not a directory: {whisper_path}"
            )

    def _configure_offline_environment(self):

        os.environ["HF_HOME"] = settings.hf_home

        os.environ["HF_HUB_OFFLINE"] = (
            "1" if settings.hf_hub_offline else "0"
        )

        os.environ["TRANSFORMERS_OFFLINE"] = (
            "1" if settings.transformers_offline else "0"
        )

    def _validate_device(self):

        if self.device.startswith("cuda"):

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is configured but CUDA is not available."
                )

            device = torch.device(self.device)

            index = device.index

            if index is None:
                index = 0

            device_count = torch.cuda.device_count()

            if index >= device_count:
                raise RuntimeError(
                    f"CUDA device {index} does not exist. "
                    f"Available devices: {device_count}"
                )

            print(
                f"[OmniVoice] CUDA device: "
                f"{torch.cuda.get_device_name(index)}"
            )

    def load(self):

        if self.loaded:
            print("[OmniVoice] Model already loaded.")
            return

        print("[OmniVoice] Starting model initialization...")

        self._configure_offline_environment()
        self._validate_paths()
        self._validate_device()

        print(
            f"[OmniVoice] Device: {self.device}"
        )

        print(
            f"[OmniVoice] Dtype: {self.dtype}"
        )

        print(
            "[OmniVoice] Loading OmniVoice..."
        )

        # Import only after offline environment is configured.
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            settings.omnivoice_model_path,
            device_map=self.device,
            dtype=self.dtype,
            local_files_only=True,
        )

        print(
            "[OmniVoice] Loading Whisper ASR..."
        )

        OmniVoice.load_asr_model(
            self.model,
            model_name=settings.whisper_model_path,
        )

        self.loaded = True

        print("[OmniVoice] Model is READY.")

    def generate(
        self,
        text: str,
        ref_audio: str,
        language: str,
        ref_text: str | None = None,
    ):

        if not self.loaded or self.model is None:
            raise RuntimeError(
                "OmniVoice model is not loaded."
            )

        return self.model.generate(
            text=text,
            ref_audio=ref_audio,
            language=language,
            ref_text=ref_text,
        )

    def unload(self):

        if self.model is None:
            return

        print("[OmniVoice] Unloading model...")

        del self.model
        self.model = None

        self.loaded = False

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OmniVoice] Model unloaded.")

    def status(self):

        return {
            "loaded": self.loaded,
            "device": self.device,
            "dtype": str(self.dtype),
        }