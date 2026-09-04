from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "rag-learning-lab")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 100)


settings = Settings()


def ensure_upload_dir() -> Path:
    upload_path = BASE_DIR / settings.upload_dir
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path
