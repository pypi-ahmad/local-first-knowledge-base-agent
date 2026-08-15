"""Local audio transcription via faster-whisper (CPU by default).

ponytail: CPU int8 inference is the local-first default; slow on long files.
Set WHISPER_DEVICE=cuda in .env if the machine has a supported GPU.
"""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE

_model_cache: dict[str, WhisperModel] = {}


def _get_model(model_size: str) -> WhisperModel:
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(model_size, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return _model_cache[model_size]


def transcribe_audio(path: Path, model_size: str = "") -> str:
    model = _get_model(model_size or WHISPER_MODEL_SIZE)
    segments, _info = model.transcribe(str(path))
    return " ".join(segment.text.strip() for segment in segments)
