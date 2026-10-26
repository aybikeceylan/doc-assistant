"""Uçtan uca RAG: soru -> arama -> prompt -> cevap.

Çalıştır:
    python ask.py "Kuzey Teknoloji'de yıllık izin kaç gün?"
    python ask.py --show-prompt "Eğitim bütçesi ne kadar?"
    python ask.py                 # soru sormaya devam eden etkileşimli mod
"""

import argparse

from rag.config import INDEX_PATH, get_client
from rag.embedder import embed
from rag.generator import build_prompt, generate_answer
from rag.store import VectorStore


def answer(question: str, top_k: int, show_prompt: bool) -> None:
    query_vector = embed(client, [question], task_type="RETRIEVAL_QUERY")[0]
    results = store.search(query_vector, top_k=top_k)

    if show_prompt:
        print("----- MODELE GİDEN PROMPT -----")
        print(build_prompt(question, results))
        print("-------------------------------\n")

    print(generate_answer(client, question, results))
    sources = ", ".join(f"{r.chunk.id} ({r.score:.2f})" for r in results)
    print(f"\n(getirilen parçalar: {sources})\n")


parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="?")
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--show-prompt", action="store_true")
args = parser.parse_args()

client = get_client()
store = VectorStore.load(INDEX_PATH)

if args.question:
    answer(args.question, args.top_k, args.show_prompt)
else:
    print("Soru sor (çıkmak için boş bırakıp Enter):")
    while question := input("> ").strip():
        answer(question, args.top_k, args.show_prompt)
