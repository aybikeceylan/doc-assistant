"""Belgeleri parçala, embedding'lerini çıkar ve index.json'a kaydet.

Belgeler veya chunk ayarları değiştiğinde tekrar çalıştır.

Çalıştır:
    python ingest.py
    python ingest.py 200 40    # chunk_size=200, overlap=40
"""

import sys

from rag.chunker import load_and_chunk
from rag.config import DATA_DIR, INDEX_PATH, get_client
from rag.embedder import embed
from rag.store import VectorStore

chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 400
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 80

chunks = load_and_chunk(DATA_DIR, chunk_size, overlap)
print(f"{len(chunks)} parça oluşturuldu (chunk_size={chunk_size}, overlap={overlap}).")

vectors = embed(get_client(), [c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
print(f"Embedding'ler alındı: {vectors.shape}")

VectorStore(chunks, vectors).save(INDEX_PATH)
print(f"Kaydedildi: {INDEX_PATH.name}")
