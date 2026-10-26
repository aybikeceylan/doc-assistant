"""Task 3: Embedding = anlamı sayılara çevirmek.

Aynı anlamdaki cümleler, kelimeleri farklı olsa bile birbirine yakın vektörler üretir.

Çalıştır: python -m steps.step3_embeddings
"""

from rag.config import get_client
from rag.embedder import cosine_similarity, embed

sentences = [
    "Yıllık izin hakkım kaç gün?",
    "Bir yılda kaç gün tatil yapabilirim?",
    "Şifrem kaç karakter olmalı?",
    "Kedimin mamasını değiştirmeli miyim?",
]

client = get_client()
vectors = embed(client, sentences)
print(f"Her cümle {vectors.shape[1]} boyutlu bir vektör oldu.")
print(f"İlk vektörün ilk 5 sayısı: {vectors[0][:5].round(4)}\n")

base = sentences[0]
print(f'"{base}" cümlesine benzerlik:')
for sentence, vector in zip(sentences[1:], vectors[1:]):
    score = cosine_similarity(vectors[0], vector)
    print(f"  {score:.3f}  {sentence}")
