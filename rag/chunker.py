from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Metni en fazla `chunk_size` karakterlik, `overlap` karakter örtüşen parçalara böler.

    Kelimeyi ortadan kesmemek için parça sınırını en yakın boşluğa çeker.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap, chunk_size'dan küçük olmalı")

    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def load_and_chunk(data_dir: Path, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    chunks = []
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(id=f"{path.stem}#{i}", source=path.name, text=piece))
    return chunks
