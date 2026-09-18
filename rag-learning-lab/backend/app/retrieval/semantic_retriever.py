from __future__ import annotations

from typing import Any

from app.embeddings.base import EmbeddingProvider
from app.retrieval.base import RetrieverBase
from app.vectorstore.chroma_store import SimpleVectorStore


class SemanticRetriever(RetrieverBase):
    """Semantic search retriever using vector embeddings and cosine similarity.

    This retriever embeds the query and searches the vector store for documents
    with the highest cosine similarity to the query embedding.
    """

    def __init__(
        self,
        vector_store: SimpleVectorStore,
        embedding_provider: EmbeddingProvider,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search for documents using semantic similarity.

        Args:
            query: The search query (will be embedded)
            top_k: Number of top results to return

        Returns:
            List of result dicts with semantic similarity scores
        """
        query_vector = self.embedding_provider.embed_query(query)
        results = self.vector_store.query(query_vector=query_vector, top_k=top_k)

        # Wrap results with consistent format
        wrapped = []
        for result in results:
            wrapped.append(
                {
                    **result,
                    "retrieval_score": result.get("similarity_score"),
                    "retrieval_method": "semantic",
                }
            )
        return wrapped

    def index(self, items: list[dict[str, Any]]) -> None:
        """Index documents by embedding and adding to vector store.

        Args:
            items: List of chunk dicts with 'chunk_id', 'text', 'metadata'
        """
        if not items:
            return

        # Extract texts and embed
        texts = [item.get("text", "") for item in items]
        vectors = self.embedding_provider.embed_documents(texts)

        # Add to vector store
        for item, vector in zip(items, vectors):
            self.vector_store.add(
                chunk_id=str(item.get("chunk_id", "")),
                text=str(item.get("text", "")),
                vector=vector,
                metadata=item.get("metadata", {}),
            )

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks for a document.

        Args:
            document_id: The document to remove
        """
        self.vector_store.delete_by_document_id(document_id)
