# RAG Öğrenme Rehberi

Her task'ı sırayla yap: komutu çalıştır, çıktıya bak, "Dene" kısmını uygula, sonra kavram açıklamasını oku.
Tüm komutlar proje klasöründe ve sanal ortam aktifken çalıştırılır:

```bash
cd ~/Projects/doc-assistant
source .venv/bin/activate
```

---

## Task 0 - Kurulum

1. [Google AI Studio](https://aistudio.google.com/apikey) üzerinden ücretsiz bir API key al.
2. `cp .env.example .env` ve `.env` içindeki `GEMINI_API_KEY` değerini doldur.
3. `python -m steps.step0_hello`

**Kavram:** LLM'e bir API üzerinden metin gönderip metin alıyoruz. Model bir "kara kutu"; sadece eğitildiği verileri biliyor.

---

## Task 1 - Problem: RAG olmadan

`python -m steps.step1_without_rag`

Model, `data/` klasöründeki kurgusal Kuzey Teknoloji belgelerini hiç görmedi. Ya "bilmiyorum" der ya da makul görünen ama yanlış bir sayı uydurur (**halüsinasyon**).

**Kavram:** LLM'in bilgisi eğitim tarihinde donmuştur ve senin özel belgelerini içermez. Çözüm: soruyla ilgili bilgiyi bulup **prompt'un içine koymak**. RAG = Retrieval-Augmented Generation = "bulunan bilgiyle zenginleştirilmiş üretim".

**Dene:** `data/leave_policy.md` dosyasını aç, gerçek cevabı (24 gün) gör.

---

## Task 2 - Chunking (parçalama)

`python -m steps.step2_chunking` ve sonra `python -m steps.step2_chunking 150 30`

**Kavram:** Belgelerin tamamını her soruda modele gönderemeyiz (pahalı, yavaş, gürültülü). Belgeleri küçük parçalara bölüp sadece ilgili parçaları göndeririz.
- `chunk_size`: Parça çok küçükse bağlam kopar ("24 gün" ile "yıllık izin" farklı parçalara düşebilir). Çok büyükse aramada konu karışır.
- `overlap`: Parçaların uçları örtüşür, böylece sınırda kalan cümle iki parçada da bulunur. Çıktıda bir parçanın sonunun bir sonrakinin başında tekrar ettiğini gör.

**Dene:** `python -m steps.step2_chunking 1000 0` ile parça sayısının nasıl düştüğüne bak.

---

## Task 3 - Embedding

`python -m steps.step3_embeddings`

**Kavram:** Embedding modeli bir metni sabit uzunlukta bir sayı dizisine (vektör) çevirir. Anlamca benzer metinler birbirine yakın vektörler üretir. Yakınlığı **cosine similarity** ile ölçeriz (1'e yakın = çok benzer).

"Yıllık izin hakkım kaç gün?" ile "Bir yılda kaç gün tatil yapabilirim?" ortak kelime neredeyse içermez ama skorları yüksek çıkar. Kelime aramasının yapamadığı şey bu: **anlamsal arama**.

**Dene:** `sentences` listesine kendi cümlelerini ekle.

---

## Task 4 - Index ve arama (Retrieval)

```bash
python ingest.py
python -m steps.step4_search "Doğum günümde izinli miyim?"
```

**Kavram:** `ingest.py` bir kez çalışır: tüm parçaların vektörlerini hesaplayıp `index.json` dosyasına yazar (bizim mini "vektör veritabanımız"). Soru gelince sadece sorunun vektörü hesaplanır ve en yüksek skorlu `top_k` parça getirilir (`rag/store.py` içindeki `search`).

Belgeler `RETRIEVAL_DOCUMENT`, sorular `RETRIEVAL_QUERY` görev tipiyle embed edilir; model ikisini birbirine göre optimize eder.

**Dene:** Farklı sorular sor; hangi belgeden parça geldiğine ve skorlara bak.

---

## Task 5 - Üretim (Generation), uçtan uca RAG

```bash
python ask.py --show-prompt "Kuzey Teknoloji'de yıllık izin kaç gün?"
python ask.py
```

**Kavram:** Bulunan parçalar ve soru tek bir prompt'ta birleşir (`rag/generator.py`). Sistem prompt'u modele "sadece bağlamı kullan, kaynak göster" der. Model artık bilgiyi **ezberden değil, önüne konan metinden** okur.

**Dene:** Task 1'deki soruları sor ve cevapları karşılaştır. `--show-prompt` ile modelin tam olarak ne gördüğünü incele.

---

## Task 6 - Değerlendirme ve iyileştirme

```bash
python evaluate.py
python evaluate.py --top-k 1
```

**Kavram:** RAG iki ayrı yerde hata yapabilir. `evaluate.py` ikisini ayrı ölçer:
1. **Retrieval hatası:** Doğru parça hiç getirilmemiştir. Model ne kadar iyi olursa olsun cevap veremez.
2. **Generation hatası:** Doğru parça geldi ama model yanlış okudu ya da uydurdu.

Son soru ("CEO kim?") belgede yok. İyi bir RAG burada uydurmak yerine "Bu bilgi belgelerde yer almıyor." demeli; bunu `SYSTEM_PROMPT` sağlıyor. Ayrıca bu soruda "en iyi skor"un diğerlerinden düşük olduğuna dikkat et.

**Deneyler:**
- `python ingest.py 120 20` ile küçük parçalar, `python evaluate.py` ile ölç. Sonra `python ingest.py 400 80` ile varsayılana dön.
- `--top-k 1` ile ve `--top-k 5` ile sonuçları karşılaştır.
- `rag/generator.py` içinde "belgelerde yer almıyor" kuralını sil, CEO sorusunda ne olduğuna bak.

---

## Task 7 - Özet ve sonraki adımlar

**Öğrendiğin akış:**

```
Belgeler -> parçala -> embed -> index.json        (ingest, bir kez)
Soru -> embed -> en yakın top_k parça -> prompt -> LLM -> kaynaklı cevap   (her soruda)
```

**Gerçek projelerde bunlar eklenir:**
- **Vektör veritabanı** (Chroma, Qdrant, pgvector): `index.json` yerine; milyonlarca parçada hızlı arama.
- **Hybrid search:** Anlamsal aramayı kelime aramasıyla (BM25) birleştirmek; ürün kodları, isimler gibi tam eşleşmelerde işe yarar.
- **Reranking:** Önce 20 parça getirip ikinci bir modelle en iyi 3'ünü seçmek.
- **Daha akıllı chunking:** Başlıklara göre bölmek, her parçanın başına belge/bölüm başlığını eklemek.
- **Arayüz:** Streamlit veya Gradio ile birkaç satırda web arayüzü.
