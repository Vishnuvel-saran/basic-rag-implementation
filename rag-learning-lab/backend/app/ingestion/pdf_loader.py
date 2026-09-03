from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz

from app.config.settings import ensure_upload_dir


def clean_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pages_from_pdf(pdf_path: str | Path) -> list[dict[str, Any]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        raw_text = page.get_text("text")
        cleaned_text = clean_extracted_text(raw_text)

        if cleaned_text:
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": cleaned_text,
                }
            )

    doc.close()
    return pages


def save_uploaded_pdf(file_obj: Any, filename: str) -> Path:
    upload_dir = ensure_upload_dir()
    save_path = upload_dir / filename
    with save_path.open("wb") as destination:
        destination.write(file_obj.file.read())
    return save_path


def extract_document_from_pdf(file_path: str | Path) -> dict[str, Any]:
    save_path = Path(file_path)
    pages = extract_pages_from_pdf(save_path)
    return {
        "filename": save_path.name,
        "page_count": len(pages),
        "pages": pages,
    }
