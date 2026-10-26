import numpy as np
from google import genai
from google.genai import types

from rag.config import EMBEDDING_MODEL

DIMENSIONS = 768
BATCH_SIZE = 100


def embed(
    client: genai.Client, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> np.ndarray:
    """Metinleri vektöre çevirir. Belgeler için RETRIEVAL_DOCUMENT, sorular için RETRIEVAL_QUERY kullan."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[i : i + BATCH_SIZE],
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIMENSIONS),
        )
        vectors.extend(e.values for e in result.embeddings)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
