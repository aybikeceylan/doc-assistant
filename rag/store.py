import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rag.chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    """En basit vektör veritabanı: parçalar + vektörler bellekte, diskte tek bir JSON dosyası."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors

    def save(self, path: Path) -> None:
        data = [
            {**asdict(chunk), "vector": vector.tolist()}
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not path.exists():
            raise SystemExit("index.json yok. Önce `python ingest.py` çalıştır.")
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(id=d["id"], source=d["source"], text=d["text"]) for d in data]
        vectors = np.array([d["vector"] for d in data], dtype=np.float32)
        return cls(chunks, vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        # Tüm parçalarla cosine similarity'yi tek seferde hesapla (matris çarpımı).
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        scores = self.vectors @ query_vector / norms
        best = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in best]
