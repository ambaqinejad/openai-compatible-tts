from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ============================================================
    # Application
    # ============================================================

    app_name: str = "OmniVoice API"
    app_version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 8000

    debug: bool = False


    # ============================================================
    # Hugging Face / Offline
    # ============================================================

    hf_home: str

    hf_hub_offline: bool = True
    transformers_offline: bool = True


    # ============================================================
    # Models
    # ============================================================

    omnivoice_model_path: str
    whisper_model_path: str


    # ============================================================
    # Hardware
    # ============================================================

    device: str = "cuda:0"
    dtype: str = "float16"


    # ============================================================
    # Audio
    # ============================================================

    output_sample_rate: int = 24000
    default_voice: str = "default"
    mp3_bitrate: str = "192k"

    # ============================================================
    # Text processing
    # ============================================================

    max_input_characters: int = 100_000
    max_chunk_characters: int = 600

    max_queue_size: int = 50

    # ============================================================
    # Config
    # ============================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
