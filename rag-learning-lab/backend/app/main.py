from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config.settings import settings
from app.ingestion.pdf_loader import extract_document_from_pdf, save_uploaded_pdf

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    saved_pdf = save_uploaded_pdf(file, file.filename)
    extracted_document = extract_document_from_pdf(saved_pdf)

    return {
        "message": "PDF uploaded and text extracted successfully.",
        "filename": extracted_document["filename"],
        "page_count": extracted_document["page_count"],
        "pages": extracted_document["pages"],
    }
