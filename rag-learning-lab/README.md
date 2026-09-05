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

### Stage 4: Vector storage and retrieval

- Store text, vectors, and metadata together
- Use similarity search to find the most relevant chunks
- Understand what a collection, vector, and Top-K result mean in practice

### Stage 5: Prompt construction and grounded answering

- Build a final prompt using retrieved chunks
- Tell the LLM to answer only from the supplied context
- Instruct it to say when the answer is missing
- Preserve source references when possible

---

## Current stage

Stage 8: Backend RAG pipeline complete; frontend next.

The backend now supports the complete learning flow:

```text
PDF upload
       -> extraction
       -> chunking
       -> OpenRouter embeddings
       -> in-memory vector retrieval
       -> grounded prompt
       -> OpenRouter LLM answer
```

The next stage is a thin frontend that calls the existing upload and query APIs.

### Current models

- Embeddings: `openai/text-embedding-3-small` through the OpenRouter embeddings API
- LLM: the model configured by `LLM_MODEL` through the OpenRouter chat completions API
- Vector store: in-memory Chroma-style store for learning; vectors are not persisted across restarts

Document and query embeddings use the same embedding model. The LLM is independent and can be changed without changing the embedding model.

### What this stage teaches

- Why the LLM should not receive the entire PDF
- Why retrieval context must be assembled deliberately
- How to create a grounded prompt from Top-K results
- Why source attribution helps trust and debugging
- Why strict instructions reduce hallucination

### Architecture in this stage

```text
User question
   + retrieved chunks
   + instruction set
       ↓
prompt builder
       ↓
LLM
       ↓
grounded answer
```

### Why this matters

The retrieval system finds the likely relevant chunks, but the LLM still needs a clear instruction set. The prompt tells it:

- use the provided context only
- do not hallucinate
- say when the information is not present
- reference supporting source material when available

This is a critical part of making RAG actually useful.

---

## Learning notes for this stage

### What is a prompt in RAG?
A prompt is the final message sent to the model. It usually contains:

- the user question
- the relevant retrieved chunks
- explicit instructions to answer only from the context

### Why not send the whole PDF?
The whole PDF is often too noisy and too large. Retrieval narrows the context to the relevant evidence. This reduces cost, improves latency, and makes response quality more grounded.

### Why say "do not invent information"?
LLMs are trained to be helpful. Without constraints, they may fill gaps with confident but wrong answers. A grounded prompt reduces this by telling the model to rely on the retrieved context and explicitly admit uncertainty.

### Why include sources?
Source references help:

- debug retrieval quality
- show the user where the answer came from
- make the system more transparent and trustworthy

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

Ask a question:

```bash
curl -X POST "http://localhost:8000/query" \
       -H "Content-Type: application/json" \
       -d '{"question":"What is this document about?","top_k":3}'
```

The API response returns the answer and compact retrieved chunk details. Each chunk includes its `chunk_id`, `context`, and `metadata`; embedding vectors remain internal and are not returned.

---

## Stage 5 code structure

```text
backend/app/rag/
  prompt_builder.py
```

This stage teaches the exact idea that the model should answer from retrieved context rather than from a fuzzy notion of the whole document.

---

## Current verification status

The chunking, embedding, and vector-store behavior are already validated. The prompt builder now has a dedicated test to confirm it includes the question, context, and source references.

---

## Next stage

Build the frontend on top of the existing backend APIs. Later learning stages can replace the in-memory vector store with persistent ChromaDB and compare additional embedding providers.
