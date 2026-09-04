# RAG Learning Lab

This project is being built incrementally to teach the fundamentals of a Retrieval-Augmented Generation (RAG) system without hiding the important concepts behind frameworks.

## Stage progression

### Stage 1: PDF extraction

- Upload a PDF
- Save it to disk
- Extract page text with PyMuPDF
- Clean the raw text before further processing

### Stage 2: Chunking

- Split long text into fixed-size chunks
- Add overlap between chunks
- Keep page and chunk metadata for traceability

### Stage 3: Embeddings

- Convert text chunks into numerical vectors
- Separate embedding responsibilities from LLM responsibilities
- Prepare the data for semantic retrieval in the next stage

---

## Current stage

Stage 3: Embeddings.

### What this stage teaches

- What an embedding is
- Why embeddings are useful for semantic search
- Why we use a separate embedding layer from the LLM layer
- Why the same embedding model should generally be used consistently for stored documents and queries
- Why the embedding model is not the same thing as the generation model

### Architecture in this stage

- Text chunks are passed to an embedding provider
- The provider converts text into vectors
- The query text is also embedded into a vector
- Those vectors are later compared in the vector store for similarity search

### Important distinction

Embedding model:
- converts text into vectors

LLM:
- consumes context + question and generates an answer

These are different jobs.

### Why this matters

If embeddings are poor or inconsistent:

- retrieval quality is weak
- similar meaning may not be picked up
- the LLM may receive irrelevant context
- the answer may be grounded in the wrong chunks

---

## Learning notes for this stage

### What is an embedding?
An embedding is a dense numerical vector that represents the meaning of text in a high-dimensional space.

### Why embeddings help
Similar ideas tend to produce vectors that are close together in embedding space. This allows a semantic search system to match questions with relevant document chunks even when the words are not identical.

### Why keyword matching is different
Keyword search matches exact words or phrase overlap. Semantic search tries to measure meaning similarity between vectors. This is why a question can retrieve a relevant passage even if it does not use the same exact wording.

### Why provider abstraction matters
The retrieval pipeline should depend on an interface like:

```python
class EmbeddingProvider:
    def embed_documents(self, texts):
        pass

    def embed_query(self, text):
        pass
```

This means the application can later switch between embedding providers without rewriting the retrieval logic.

---

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then upload a PDF with a request like:

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@sample.pdf"
```

---

## Stage 3 code structure

```text
backend/app/embeddings/
  base.py
  factory.py
```

The application now includes a simple local embedding provider stub that demonstrates the abstraction and teaches the interface shape before a real provider is added.

---

## Current verification status

The chunking tests were verified, and the embedding interface is tested through a simple provider contract check.

---

## Next stage

After embeddings, the next step is vector storage and semantic retrieval using similarity search.
