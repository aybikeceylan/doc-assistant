"""Task 4: Sadece arama (retrieval). LLM yok; soruya en yakın parçaları görüyoruz.

Önce: python ingest.py
Çalıştır: python -m steps.step4_search "Doğum günümde izinli miyim?"
"""

import sys

from rag.config import INDEX_PATH, get_client
from rag.embedder import embed
from rag.store import VectorStore

question = sys.argv[1] if len(sys.argv) > 1 else "Doğum günümde izinli miyim?"

store = VectorStore.load(INDEX_PATH)
query_vector = embed(get_client(), [question], task_type="RETRIEVAL_QUERY")[0]

print(f"SORU: {question}\n")
for rank, result in enumerate(store.search(query_vector, top_k=3), start=1):
    print(f"{rank}. [{result.chunk.id}] skor={result.score:.3f}")
    print(f"   {result.chunk.text}\n")
