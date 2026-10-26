"""Task 1: RAG olmadan soru sor.

Model, Kuzey Teknoloji'nin kurgusal belgelerini hiç görmedi.
Ya "bilmiyorum" diyecek ya da (daha kötüsü) uyduracak. RAG'in çözdüğü problem bu.

Çalıştır: python -m steps.step1_without_rag
"""

from rag.config import CHAT_MODEL, get_client

QUESTIONS = [
    "Kuzey Teknoloji'de yıllık izin kaç gün?",
    "Kuzey Teknoloji'de eğitim bütçesi ne kadar?",
]

client = get_client()
for question in QUESTIONS:
    response = client.models.generate_content(model=CHAT_MODEL, contents=question)
    print(f"SORU : {question}")
    print(f"CEVAP: {response.text.strip()}\n")
