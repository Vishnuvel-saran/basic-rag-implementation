from __future__ import annotations

from app.config.settings import settings
from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider
from app.rag.prompt_builder import PromptBuilder
from app.retrieval.base import RetrieverBase
from app.retrieval.bm25_index import BM25Index
from app.retrieval.retriever_factory import create_retriever
from app.vectorstore.chroma_store import SimpleVectorStore


class RAGPipeline:
    """Minimal end-to-end RAG pipeline for learning.

    The design stays deliberately simple:
    - chunk and store documents
    - embed queries and documents
    - retrieve relevant chunks using the selected retrieval method
    - assemble a grounded prompt with the retrieved results
    - generate an answer through a provider abstraction
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: SimpleVectorStore,
        llm_provider: LLMProvider,
        top_k: int | None = None,
        retrieval_method: str | None = None,
        rrf_k_constant: int | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.bm25_index = BM25Index()
        self.top_k = top_k if top_k is not None else settings.top_k

        # Initialize retriever based on method
        method = retrieval_method or settings.retrieval_method
        rrf_k = rrf_k_constant or settings.rrf_k_constant
        self.retriever = create_retriever(
            method=method,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            bm25_index=self.bm25_index,
            rrf_k_constant=rrf_k,
        )
        self.retrieval_method = method

    def index_chunks(self, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        document_ids = {str(chunk.get("document_id", "unknown")) for chunk in chunks}
        for document_id in document_ids:
            self.vector_store.delete_by_document_id(document_id)

        texts = [chunk.get("chunk_text", "") for chunk in chunks]
        document_vectors = self.embedding_provider.embed_documents(texts)

        stored_items: list[dict] = []
        for chunk, vector in zip(chunks, document_vectors):
            metadata = {
                "document_id": chunk.get("document_id", "unknown"),
                "filename": chunk.get("filename", "unknown"),
                "page_number": chunk.get("page_number", 0),
                "page_ids": chunk.get("page_ids", [chunk.get("page_number", 0)]),
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "chunking_strategy": chunk.get("chunking_strategy", "fixed"),
                "chunk_index": chunk.get("chunk_index", 0),
                "chunk_size": chunk.get("chunk_size"),
            }
            self.vector_store.add(
                chunk_id=str(metadata["chunk_id"]),
                text=str(chunk.get("chunk_text", "")),
                vector=vector,
                metadata=metadata,
            )
            stored_items.append({**chunk, "metadata": metadata})

        # Rebuild BM25 index from current vector store state to keep both in sync
        self.bm25_index.build(self.vector_store._items)

        return stored_items

    def query(
        self,
        question: str,
        top_k: int | None = None,
        retrieval_method: str | None = None,
        llm_options: dict | None = None,
        include_bm25: bool = False,
    ) -> dict:
        effective_top_k = self.top_k if top_k is None else top_k

        # Use specified retrieval method or fall back to instance default
        method = retrieval_method or self.retrieval_method

        # Support legacy include_bm25 flag for backward compatibility
        if include_bm25 and method == "semantic":
            # For backward compat, include_bm25=True with semantic means also get BM25 results
            pass

        # Use the retriever to get results
        retrieved_chunks = self.retriever.search(question, top_k=effective_top_k)

        # For backward compatibility, still add BM25 results if requested
        retrieved_chunks_bm25 = None
        if include_bm25 and method != "bm25":
            bm25_results = self.bm25_index.search(question, top_k=effective_top_k)
            retrieved_chunks_bm25 = [
                {**result, "retrieval_method": "bm25"} for result in bm25_results
            ]

        prompt = PromptBuilder.build(question, retrieved_chunks, top_k=effective_top_k)
        llm_options = {
            "temperature": settings.llm_temperature,
            "max_output_tokens": settings.llm_max_output_tokens,
            **(llm_options or {}),
        }
        if hasattr(self.llm_provider, "generate_with_metadata"):
            llm_result = self.llm_provider.generate_with_metadata(prompt, **llm_options)
            answer = llm_result["answer"]
        else:
            answer = self.llm_provider.generate(prompt, **llm_options)
            llm_result = {"answer": answer, "usage": None}

        result = {
            "question": question,
            "top_k": effective_top_k,
            "retrieval_method": method,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "prompt": prompt,
            "llm": llm_result,
        }

        # Include BM25 results if requested (backward compatibility)
        if include_bm25 and retrieved_chunks_bm25 is not None:
            result["retrieved_chunks_bm25"] = retrieved_chunks_bm25

        return result
