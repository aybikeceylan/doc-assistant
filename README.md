# doc-assistant

Kurgusal "Kuzey Teknoloji" şirketinin çalışan el kitabına soru cevaplayan, sıfırdan yazılmış küçük bir RAG projesi (Gemini API + numpy).

## Kurulum

```bash
cd ~/Projects/doc-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY değerini doldur
```

## Kullanım

```bash
python ingest.py                         # belgeleri indexle
python ask.py "Yıllık izin kaç gün?"     # soru sor
python evaluate.py                       # mini test seti
```

Adım adım öğrenmek için [NOTES.md](NOTES.md) dosyasını takip et.

## Yapı

- `data/` - kurgusal belgeler
- `rag/chunker.py` - parçalama
- `rag/embedder.py` - embedding + cosine similarity
- `rag/store.py` - vektör index'i ve arama
- `rag/generator.py` - prompt + cevap üretimi
- `steps/` - her kavramı tek başına gösteren küçük scriptler
