from app.audio.chunker import chunk_text


text = """
سلام. این یک متن آزمایشی است.

در این پروژه قرار است متن‌های طولانی را به قسمت‌های مناسب تقسیم کنیم.
هر قسمت به صورت جداگانه به OmniVoice داده می‌شود.
سپس صداهای تولید شده به یکدیگر متصل خواهند شد.

این روش باعث می‌شود محدودیت طول ورودی مدل مشکل‌ساز نشود.
"""


chunks = chunk_text(
    text,
    max_length=100,
)


for i, chunk in enumerate(chunks, 1):

    print("=" * 60)
    print(f"CHUNK {i}")
    print("=" * 60)
    print(chunk)
    print(f"Length: {len(chunk)}")