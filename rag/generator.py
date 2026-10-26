from google import genai
from google.genai import types

from rag.config import CHAT_MODEL
from rag.store import SearchResult

SYSTEM_PROMPT = """Sen Kuzey Teknoloji çalışanlarına yardım eden bir asistansın.
Kurallar:
- Sadece verilen BAĞLAM içindeki bilgileri kullan, kendi bilgini ekleme.
- Cevap bağlamda yoksa sadece "Bu bilgi belgelerde yer almıyor." de.
- Kullandığın her bilginin sonuna kaynağını köşeli parantezle yaz, örn: [leave_policy#0].
- Kısa ve net cevap ver."""


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(f"[{r.chunk.id}]\n{r.chunk.text}" for r in results)
    return f"BAĞLAM:\n{context}\n\nSORU: {question}"


def generate_answer(client: genai.Client, question: str, results: list[SearchResult]) -> str:
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=build_prompt(question, results),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1),
    )
    return response.text.strip()
