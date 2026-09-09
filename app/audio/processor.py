import numpy as np


def concatenate_audio(
    audio_chunks: list[np.ndarray],
    silence_ms: int = 80,
    sample_rate: int = 24000,
) -> np.ndarray:

    if not audio_chunks:
        raise ValueError(
            "No audio chunks to concatenate."
        )

    processed = []

    silence_samples = int(
        sample_rate * silence_ms / 1000
    )

    silence = np.zeros(
        silence_samples,
        dtype=np.float32,
    )

    for index, audio in enumerate(audio_chunks):

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.ndim > 1:
            audio = np.squeeze(audio)

        processed.append(audio)

        if index < len(audio_chunks) - 1:
            processed.append(silence)

    return np.concatenate(processed)