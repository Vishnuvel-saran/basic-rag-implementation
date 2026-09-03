from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "rag-learning-lab")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")


settings = Settings()


def ensure_upload_dir() -> Path:
    upload_path = BASE_DIR / settings.upload_dir
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path
