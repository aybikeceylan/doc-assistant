"""Task 6: Mini test seti ile RAG'i ölç.

İki şeyi ayrı ayrı kontrol eder:
  1. Retrieval: Doğru belge, getirilen parçalar arasında mı?
  2. Cevap: Beklenen bilgi (anahtar kelime) cevapta geçiyor mu?

Çalıştır:
    python evaluate.py
    python evaluate.py --top-k 1
"""

import argparse

from rag.config import INDEX_PATH, get_client
from rag.embedder import embed
from rag.generator import generate_answer
from rag.store import VectorStore

# (soru, beklenen kaynak dosya, cevapta geçmesi beklenen ifade)
TEST_SET = [
    ("Yıllık izin kaç gün?", "leave_policy.md", "24"),
    ("Doğum günümde çalışmak zorunda mıyım?", "leave_policy.md", "doğum gün"),
    ("Kişisel gelişim için ne kadar para harcayabilirim?", "benefits.md", "25.000"),
    ("Hangi gün herkes ofiste olmalı?", "remote_work.md", "Salı"),
    ("Şifrem en az kaç karakter olmalı?", "it_rules.md", "14"),
    ("Şirketin CEO'su kim?", None, "belgelerde yer almıyor"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--top-k", type=int, default=3)
args = parser.parse_args()

client = get_client()
store = VectorStore.load(INDEX_PATH)

retrieval_hits = answer_hits = 0
for question, expected_source, expected_text in TEST_SET:
    query_vector = embed(client, [question], task_type="RETRIEVAL_QUERY")[0]
    results = store.search(query_vector, top_k=args.top_k)
    answer = generate_answer(client, question, results)

    sources = {r.chunk.source for r in results}
    retrieval_ok = expected_source is None or expected_source in sources
    answer_ok = expected_text.lower() in answer.lower()
    retrieval_hits += retrieval_ok
    answer_hits += answer_ok

    print(f"SORU : {question}")
    print(f"CEVAP: {answer}")
    print(f"  retrieval {'OK ' if retrieval_ok else 'HATA'} | cevap {'OK' if answer_ok else 'HATA'}"
          f" | en iyi skor {results[0].score:.3f}\n")

total = len(TEST_SET)
print(f"Retrieval: {retrieval_hits}/{total}   Cevap: {answer_hits}/{total}   (top_k={args.top_k})")
