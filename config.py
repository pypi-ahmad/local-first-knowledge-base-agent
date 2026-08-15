"""Environment, paths, and model catalog. Single source of truth for config."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "db" / "data"
CHROMA_DIR = DATA_DIR / "chroma"
METADATA_DB_PATH = DATA_DIR / "metadata.sqlite3"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_COLLECTION = "knowledge_base"

# --- Providers -------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_COMPAT_MODELS = ["gpt-5.6-luna", "gpt-5.6-terra"]

AGNES_API_KEY = os.environ.get("AGNES_API_KEY")
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL = "agnes-2.5-flash"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.7-flash"]

# Models that take a reasoning_effort parameter (OpenAI-compatible reasoning models).
REASONING_EFFORT_MODELS = {"gpt-5.6-luna", "gpt-5.6-terra"}
REASONING_EFFORT = "medium"

PROVIDERS = ["ollama", "openai_compatible", "agnes", "gemini"]

# --- Indexing sources --------------------------------------------------------

SUPPORTED_TEXT_EXT = {".md", ".markdown", ".txt"}
SUPPORTED_PDF_EXT = {".pdf"}
SUPPORTED_CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".cs", ".sh", ".ps1", ".sql", ".json", ".yaml", ".yml",
}
ALL_INDEXABLE_EXT = SUPPORTED_TEXT_EXT | SUPPORTED_PDF_EXT | SUPPORTED_CODE_EXT

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Default Chromium browser History file locations on Windows.
BROWSER_HISTORY_PATHS = {
    "Chrome": Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data" / "Default" / "History",
    "Edge": Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Edge" / "User Data" / "Default" / "History",
}

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 6
GRAPH_HOP_DEPTH = 2
QUERY_EXPANSION_COUNT = 3
HYBRID_VECTOR_WEIGHT = 0.6  # remainder goes to BM25 keyword score

# --- Multi-modal ------------------------------------------------------------

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
ALL_INDEXABLE_EXT = ALL_INDEXABLE_EXT | SUPPORTED_IMAGE_EXT | SUPPORTED_AUDIO_EXT

# Vision-capable Ollama model used for image OCR + captioning. Override via env
# if the user's pulled model differs (e.g. "qwen3-vl:4b", "minicpm-v4.5").
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl:4b")

# faster-whisper model size for local audio transcription. Bigger = more accurate,
# slower on CPU. "small" is a reasonable local-first default.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# --- Privacy -----------------------------------------------------------------

# When true, only the Ollama provider may be used; remote API calls are refused.
LOCAL_ONLY_MODE_DEFAULT = os.environ.get("LOCAL_ONLY_MODE", "false").lower() == "true"

# --- Pricing (USD per 1M tokens) — for the estimated-cost display next to the
# model picker. Not billing-accurate; a rough guide only. Update as rates change.
PRICING_USD_PER_1M = {
    "sonnet-5": {"input": 2.00, "output": 10.00},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "grok-4.6": {"input": 2.00, "output": 6.00},
    "agnes-2.5-flash": {"input": 0.0, "output": 0.0},
}
