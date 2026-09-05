# RAG Learning Lab

A backend-first learning project for understanding Retrieval-Augmented Generation from PDF upload to grounded LLM answers.

The project deliberately implements the important RAG stages explicitly instead of hiding them behind LangChain or LlamaIndex.

## Current Status

The backend RAG pipeline is complete. The next stage is building a frontend on top of the existing API.

```text
PDF upload
   -> PDF text extraction
   -> text chunking
   -> document embeddings
   -> vector storage and similarity retrieval
   -> grounded prompt construction
   -> LLM answer
```

## Current Providers

- Embeddings: `openai/text-embedding-3-small` through the OpenRouter embeddings API
- LLM: the model configured by `LLM_MODEL` through OpenRouter chat completions
- Vector store: an in-memory Chroma-style store implemented for learning
- PDF extraction: PyMuPDF
- API: FastAPI

Document chunks and user queries use the same embedding model. The embedding model and LLM are separate components and can be changed independently.

The current vector store is in memory, so uploaded documents must be uploaded again after restarting the server. Persistent ChromaDB can be added in a later stage.

## Project Structure

```text
rag-learning-lab/
  backend/
    app/
      config/          Environment-backed settings
      ingestion/       PDF extraction and chunking
      embeddings/      Embedding provider abstraction and providers
      vectorstore/     In-memory vector storage and cosine search
      rag/             Prompt construction and end-to-end pipeline
      llm/             LLM provider abstraction and OpenRouter provider
    tests/             Backend tests
```

## Configuration

Copy the example environment file and set the required API key:

```bash
cd rag-learning-lab/backend
cp .env.example .env
```

Important settings in `.env`:

```env
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K=3

EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=your_openrouter_api_key_here

LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

`CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` are used as application defaults. A query can still provide its own `top_k` value.

Keep `.env` private and never commit API keys.

## Run Locally

From `rag-learning-lab/backend`:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Usage

Upload and index a PDF:

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F 'file=@sample.pdf'
```

Override chunking for one upload:

```bash
curl -X POST "http://localhost:8000/documents/upload?chunk_size=400&chunk_overlap=50" \
  -F 'file=@sample.pdf'
```

Ask a question:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","top_k":3}'
```

The API returns the answer, sources, and compact retrieved chunk details. Embedding vectors are used internally for similarity search and are not returned:

```json
{
  "chunk_id": "2_0",
  "context": "Retrieved chunk text...",
  "metadata": {
    "filename": "sample.pdf",
    "page_number": 2
  }
}
```

## Tests

```bash
cd rag-learning-lab/backend
python -m pytest -q
```

Current validation: all backend tests pass.

## Learning Roadmap

Completed:

1. PDF extraction
2. Fixed-size chunking with overlap
3. Embedding provider abstraction
4. OpenRouter `text-embedding-3-small` integration
5. Vector storage and cosine similarity retrieval
6. Grounded prompt construction
7. LLM provider abstraction and OpenRouter integration
8. End-to-end RAG pipeline and API endpoints

Next:

1. Build a simple frontend for upload and question answering
2. Add persistent ChromaDB storage
3. Compare embedding models and retrieval quality
4. Add evaluation examples for chunking and retrieval
