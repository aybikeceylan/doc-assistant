"""Task 0: Gemini bağlantısını test et.

Çalıştır: python -m steps.step0_hello
"""

from rag.config import CHAT_MODEL, get_client

client = get_client()
response = client.models.generate_content(
    model=CHAT_MODEL,
    contents="Merhaba! Tek cümleyle kendini tanıt.",
)
print(f"Model: {CHAT_MODEL}")
print(response.text)
