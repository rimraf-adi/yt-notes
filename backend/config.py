import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
EXPORTS_DIR = DATA_DIR / "exports"
STATIC_DIR = BASE_DIR / "frontend"

for directory in [DATA_DIR, DOWNLOADS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment
load_dotenv(BASE_DIR / ".env")

# Groq API Keys (Supports comma-separated list or individual keys)
raw_keys = os.getenv("GROQ_API_KEYS", "")
if raw_keys:
    GROQ_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
else:
    single_key = os.getenv("GROQ_API_KEY", "")
    GROQ_KEYS = [single_key.strip()] if single_key.strip() else []

# Fallback models
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek-r1-distill-llama-70b")

DB_PATH = DATA_DIR / "notebooklm.db"
