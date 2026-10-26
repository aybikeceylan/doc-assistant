"""Task 2: Belgeleri parçalara (chunk) böl ve incele. API key gerekmez.

Çalıştır:
    python -m steps.step2_chunking
    python -m steps.step2_chunking 150 30    # chunk_size=150, overlap=30
"""

import sys

from rag.chunker import load_and_chunk
from rag.config import DATA_DIR

chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 400
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 80

chunks = load_and_chunk(DATA_DIR, chunk_size, overlap)
print(f"chunk_size={chunk_size}, overlap={overlap} -> toplam {len(chunks)} parça\n")

for chunk in chunks[:4]:
    print(f"[{chunk.id}] ({len(chunk.text)} karakter)")
    print(chunk.text)
    print("-" * 60)
