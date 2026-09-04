from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.config.settings import settings
from app.ingestion.chunker import chunk_document_pages
from app.ingestion.pdf_loader import extract_document_from_pdf, save_uploaded_pdf

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int | None = Query(default=None, gt=0),
    chunk_overlap: int | None = Query(default=None, ge=0),
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    saved_pdf = save_uploaded_pdf(file, file.filename)
    extracted_document = extract_document_from_pdf(saved_pdf)

    for page in extracted_document["pages"]:
        page["document_id"] = file.filename
        page["filename"] = file.filename

    effective_chunk_size = chunk_size or settings.chunk_size
    effective_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    )

    try:
        chunks = chunk_document_pages(
            extracted_document["pages"],
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "PDF uploaded, extracted, and chunked successfully.",
    }


"""
"filename": extracted_document["filename"],
        "page_count": extracted_document["page_count"],
        "pages": extracted_document["pages"],
        "chunk_size": effective_chunk_size,
        "chunk_overlap": effective_chunk_overlap,
        "chunk_count": len(chunks),
        "chunks": chunks,
"""
