from __future__ import annotations

import time
from inspect import signature
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.embeddings.factory import create_embedding_provider
from app.ingestion.chunker import chunk_document_pages
from app.ingestion.pdf_loader import extract_document_from_pdf, save_uploaded_pdf
from app.llm.factory import create_llm_provider
from app.rag.pipeline import RAGPipeline
from app.vectorstore.chroma_store import SimpleVectorStore

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5180",
        "http://127.0.0.1:5180",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.vector_store = SimpleVectorStore()
app.state.embedding_provider = create_embedding_provider(settings.embedding_provider)
app.state.llm_provider = create_llm_provider(settings.llm_provider)
app.state.rag_pipeline = RAGPipeline(
    embedding_provider=app.state.embedding_provider,
    vector_store=app.state.vector_store,
    llm_provider=app.state.llm_provider,
)


def _public_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return useful chunk context without exposing internal embedding vectors."""
    metadata = chunk.get("metadata", {})
    public_chunk = {
        "chunk_id": chunk.get("chunk_id", metadata.get("chunk_id")),
        "context": chunk.get("text", chunk.get("chunk_text", "")),
        "metadata": metadata,
    }
    if chunk.get("similarity_score") is not None:
        public_chunk["similarity_score"] = chunk["similarity_score"]
    return public_chunk


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Query(default=settings.chunking_strategy),
    chunk_size: int | None = Query(default=None, gt=0),
    chunk_overlap: int | None = Query(default=None, ge=0),
    semantic_threshold: float | None = Query(default=None, ge=0, le=1),
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    saved_pdf = save_uploaded_pdf(file, file.filename)
    extracted_document = extract_document_from_pdf(saved_pdf)

    for page in extracted_document["pages"]:
        page["document_id"] = file.filename
        page["filename"] = file.filename

    effective_chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    effective_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    )
    effective_semantic_threshold = (
        semantic_threshold
        if semantic_threshold is not None
        else settings.semantic_threshold
    )

    try:
        chunks = chunk_document_pages(
            extracted_document["pages"],
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            strategy=chunking_strategy,
            semantic_threshold=effective_semantic_threshold,
            embedding_provider=app.state.embedding_provider,
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
        "chunking_strategy": chunking_strategy,
        "semantic_threshold": effective_semantic_threshold,
        "chunk_count": len(indexed_chunks),
        "chunks": [_public_chunk(chunk) for chunk in indexed_chunks],
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

    temperature = payload.get("temperature", settings.llm_temperature)
    max_output_tokens = payload.get("max_output_tokens", settings.llm_max_output_tokens)
    model = payload.get("model", settings.llm_model)
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise HTTPException(
            status_code=400, detail="'temperature' must be between 0 and 2."
        )
    if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        raise HTTPException(
            status_code=400, detail="'max_output_tokens' must be positive."
        )
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(
            status_code=400, detail="'model' must be a non-empty string."
        )

    started_at = time.perf_counter()
    query_options = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "model": model,
    }
    query_method = app.state.rag_pipeline.query
    if "llm_options" in signature(query_method).parameters:
        result = query_method(question, top_k=top_k, llm_options=query_options)
    else:
        result = query_method(question, top_k=top_k)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    sources = []
    for chunk in result.get("retrieved_chunks", []):
        metadata = chunk.get("metadata", {})
        sources.append(
            f"{metadata.get('filename', 'unknown')} | page {metadata.get('page_number', '?')}"
        )

    usage = result.get("llm", {}).get("usage")
    telemetry = {
        "input_tokens": usage.get("prompt_tokens") if usage else None,
        "output_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
        "token_source": "actual" if usage else "not_provided",
        "latency_ms": latency_ms,
        "estimated_cost": None,
    }

    return {
        "question": result["question"],
        "answer": result["answer"],
        "top_k": result["top_k"],
        "sources": sources,
        "retrieved_chunks": [
            _public_chunk(chunk) for chunk in result["retrieved_chunks"]
        ],
        "prompt": result["prompt"],
        "retrieval": {
            "top_k": result["top_k"],
            "similarity_metric": "cosine",
            "total_chunks_searched": len(app.state.vector_store._items),
            "embedding_model": settings.embedding_model,
            "embedding_dimension": (
                len(result["retrieved_chunks"][0]["vector"])
                if result["retrieved_chunks"]
                else None
            ),
        },
        "llm": {
            "model": result.get("llm", {}).get("model", model),
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "context_window": "provider managed",
        },
        "telemetry": telemetry,
    }
