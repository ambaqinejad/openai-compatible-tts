import re
import httpx

from app.config import settings


class TextNormalizer:
    """
    Pre-process text before sending it to TTS.

    Responsibilities:
    - Add punctuation.
    - Add Persian ezafe where useful.
    - Add limited diacritics where useful for pronunciation.
    - Preserve the original wording and meaning.
    - Do NOT summarize, translate, rewrite, or add content.
    """

    def __init__(self):
        self.base_url = settings.gemma_base_url.rstrip("/")
        self.model = settings.gemma_model
        self.timeout = settings.gemma_timeout

    def _build_prompt(self, text: str) -> str:
        return f"""
متن زیر قرار است مستقیماً به یک موتور تبدیل متن به گفتار فارسی داده شود.

وظیفه تو فقط آماده‌سازی متن برای گفتار است.

قوانین بسیار مهم:

1. کلمات، جمله‌ها و معنای متن را تغییر نده.
2. هیچ جمله یا کلمه جدیدی اضافه نکن.
3. چیزی را حذف نکن.
4. ترجمه، خلاصه‌سازی یا بازنویسی انجام نده.
5. فقط نشانه‌گذاری و اصلاحات لازم برای تلفظ را انجام بده.
6. در جاهایی که لازم است از علائم نگارشی فارسی استفاده کن:
   ،  ؛  :  ؟  !
7. در پایان جمله‌ها در صورت نیاز نقطه بگذار.
8. برای مکث‌های طبیعی از «،» استفاده کن.
9. برای اضافه، در موارد واضح و مفید از کسره اضافه استفاده کن.
   مثال:
   برادر علی
   → برادرِ علی

10. برای کلماتی که بدون اعراب ممکن است برای TTS اشتباه یا مبهم تلفظ شوند،
    فقط در صورت نیاز اعراب محدود اضافه کن.
    مثال:
    سرنوشت
    → سَرنِوشت

11. از اعراب‌گذاری کامل و غیرضروری کل متن خودداری کن.
12. از نشانه‌های خاص، Markdown، توضیح، نقل‌قول و کامنت استفاده نکن.
13. خروجی باید فقط متن نهایی آماده برای TTS باشد.

نمونه:

ورودی:
برادر علی به خانه رفت و سرنوشت خودش را تغییر داد

خروجی:
برادرِ علی به خانه رفت و سَرنِوشتِ خودش را تغییر داد.

متن اصلی:
{text}

فقط متن اصلاح‌شده را برگردان.
""".strip()

    async def normalize(self, text: str) -> str:
        text = text.strip()

        if not text:
            return text

        prompt = self._build_prompt(text)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "تو یک ویرایشگر متن فارسی برای "
                        "موتور Text-to-Speech هستی."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.0,
        }

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Gemma text normalization timed out."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Gemma text normalization failed. "
                f"HTTP {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not connect to Gemma server "
                f"at {self.base_url}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"Gemma normalization failed: {exc}"
            ) from exc

        try:
            result = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Invalid response from Gemma: {data}"
            ) from exc

        result = self._clean_output(result)

        if not result:
            raise RuntimeError(
                "Gemma returned an empty normalized text."
            )

        return result

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Remove accidental Markdown wrappers or surrounding quotes.
        Do not otherwise modify the generated text.
        """

        text = text.strip()

        # Remove markdown code fences if model accidentally returns them.
        text = re.sub(
            r"^```(?:text|plaintext|fa|persian)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

        # Remove accidental surrounding quotes.
        if (
            len(text) >= 2
            and text[0] in {'"', "«"}
            and text[-1] in {'"', "»"}
        ):
            text = text[1:-1].strip()

        return text