"""Image OCR + captioning via a local Ollama vision model.

Reuses the Ollama dependency the app already requires instead of adding a
separate OCR/captioning stack (pytesseract + Tesseract binary, or a
dedicated captioning model). Requires the user to have pulled a
vision-capable model (default config.OLLAMA_VISION_MODEL, e.g. "qwen3-vl:4b").
"""

from __future__ import annotations

from pathlib import Path

import ollama

from config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL

_PROMPT = (
    "Transcribe verbatim any visible text in this image (signs, UI, documents, "
    "code, handwriting). Then, on a new line, give a one-sentence description "
    "of the image. Output the transcription first, description second."
)


def _client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_BASE_URL)


def ocr_caption_image(path: Path, model: str = "") -> str:
    """Best-effort OCR + caption for an image file. Raises if the vision model
    is unavailable; callers should catch and skip rather than fail the whole index."""
    response = _client().chat(
        model=model or OLLAMA_VISION_MODEL,
        messages=[{"role": "user", "content": _PROMPT, "images": [str(path)]}],
    )
    return response["message"]["content"]


def ocr_caption_image_bytes(image_bytes: bytes, model: str = "") -> str:
    """Same as ocr_caption_image but for in-memory bytes (used for scanned PDF pages)."""
    response = _client().chat(
        model=model or OLLAMA_VISION_MODEL,
        messages=[{"role": "user", "content": _PROMPT, "images": [image_bytes]}],
    )
    return response["message"]["content"]
