import re


SENTENCE_ENDINGS = re.compile(
    r"(?<=[.!?؟!])\s+"
)

SEMICOLON_SPLIT = re.compile(
    r"(?<=؛)\s+"
)

COMMA_SPLIT = re.compile(
    r"(?<=،)\s+"
)


def normalize_text(text: str) -> str:
    """
    Basic text normalization before chunking.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize excessive spaces while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Prevent excessive empty lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_by_length(
    text: str,
    max_length: int,
) -> list[str]:

    chunks = []

    while len(text) > max_length:

        split_at = text.rfind(
            " ",
            0,
            max_length,
        )

        if split_at <= 0:
            split_at = max_length

        chunk = text[:split_at].strip()

        if chunk:
            chunks.append(chunk)

        text = text[split_at:].strip()

    if text:
        chunks.append(text)

    return chunks


def split_large_piece(
    text: str,
    max_length: int,
) -> list[str]:

    if len(text) <= max_length:
        return [text.strip()]

    # Try sentence boundaries.
    parts = SENTENCE_ENDINGS.split(text)

    result = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        result.extend(
            _pack_parts(
                part,
                max_length,
            )
        )

    return result


def _pack_parts(
    parts: str,
    max_length: int,
) -> list[str]:

    if len(parts) <= max_length:
        return [parts]

    result = []

    current = ""

    # Try semicolon boundaries.
    subparts = SEMICOLON_SPLIT.split(parts)

    for part in subparts:

        part = part.strip()

        if not part:
            continue

        candidate = (
            f"{current} {part}".strip()
            if current
            else part
        )

        if len(candidate) <= max_length:

            current = candidate

        else:

            if current:
                result.append(current)

            if len(part) <= max_length:
                current = part
            else:
                # Try comma boundaries.
                comma_parts = COMMA_SPLIT.split(part)

                for comma_part in comma_parts:

                    comma_part = comma_part.strip()

                    if not comma_part:
                        continue

                    candidate = (
                        f"{current} {comma_part}".strip()
                        if current
                        else comma_part
                    )

                    if len(candidate) <= max_length:
                        current = candidate

                    else:

                        if current:
                            result.append(current)

                        if len(comma_part) <= max_length:
                            current = comma_part
                        else:

                            result.extend(
                                split_by_length(
                                    comma_part,
                                    max_length,
                                )
                            )

                            current = ""

    if current:
        result.append(current)

    return result


def chunk_text(
    text: str,
    max_length: int = 600,
) -> list[str]:

    text = normalize_text(text)

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    chunks = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        chunks.extend(
            split_large_piece(
                paragraph,
                max_length,
            )
        )

    return chunks