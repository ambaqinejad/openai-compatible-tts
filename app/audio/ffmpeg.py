import subprocess
from pathlib import Path


def convert_wav_to_mp3(
    input_path: str | Path,
    output_path: str | Path,
    bitrate: str = "192k",
) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input audio does not exist: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-ar",
        "24000",
        "-ac",
        "1",
        str(output_path),
    ]

    try:

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg was not found. "
            "Make sure ffmpeg is installed "
            "and available in PATH."
        )

    except subprocess.CalledProcessError as exc:

        raise RuntimeError(
            f"FFmpeg conversion failed: "
            f"{exc.stderr.strip()}"
        )