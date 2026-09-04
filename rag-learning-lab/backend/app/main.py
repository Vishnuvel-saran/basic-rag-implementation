from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.config.settings import settings
from app.embeddings.factory import create_embedding_provider
from app.ingestion.chunker import chunk_document_pages
from app.ingestion.pdf_loader import extract_document_from_pdf, save_uploaded_pdf
from app.llm.factory import create_llm_provider
from app.rag.pipeline import RAGPipeline
from app.vectorstore.chroma_store import SimpleVectorStore

app = FastAPI(title=settings.app_name)
app.state.vector_store = SimpleVectorStore()
app.state.embedding_provider = create_embedding_provider(settings.embedding_provider)
app.state.llm_provider = create_llm_provider(settings.llm_provider)
app.state.rag_pipeline = RAGPipeline(
    embedding_provider=app.state.embedding_provider,
    vector_store=app.state.vector_store,
    llm_provider=app.state.llm_provider,
)


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

    indexed_chunks = app.state.rag_pipeline.index_chunks(chunks)

    return {
        "message": "PDF uploaded, extracted, chunked, and indexed successfully.",
        "filename": extracted_document["filename"],
        "page_count": extracted_document["page_count"],
        "pages": extracted_document["pages"],
        "chunk_size": effective_chunk_size,
        "chunk_overlap": effective_chunk_overlap,
        "chunk_count": len(indexed_chunks),
        "chunks": indexed_chunks,
    }


@app.post("/query")
async def query_document(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question")
    top_k = payload.get("top_k", settings.top_k)

    if not isinstance(question, str) or not question.strip():
        raise HTTPException(
            status_code=400, detail="A non-empty 'question' field is required."
        )

    if not isinstance(top_k, int) or top_k <= 0:
        raise HTTPException(
            status_code=400, detail="'top_k' must be a positive integer."
        )

    result = app.state.rag_pipeline.query(question, top_k=top_k)
    sources = []
    for chunk in result.get("retrieved_chunks", []):
        metadata = chunk.get("metadata", {})
        sources.append(
            f"{metadata.get('filename', 'unknown')} | page {metadata.get('page_number', '?')}"
        )

    return {
        "question": result["question"],
        "answer": result["answer"],
        "top_k": result["top_k"],
        "sources": sources,
        "retrieved_chunks": result["retrieved_chunks"],
        "prompt": result["prompt"],
    }
