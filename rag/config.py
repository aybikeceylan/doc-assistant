import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_PATH = ROOT / "index.json"

load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise SystemExit(
            "GEMINI_API_KEY bulunamadı. .env.example dosyasını .env olarak kopyalayıp key'ini yaz."
        )
    return genai.Client(api_key=api_key)
