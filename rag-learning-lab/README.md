# RAG Learning Lab

This project is being built incrementally to teach the fundamentals of a Retrieval-Augmented Generation (RAG) system without hiding the important concepts behind frameworks.

## Current stage

Stage 1: PDF extraction.

### What this stage teaches

- Why PDF extraction is the first step in a RAG pipeline
- What text extraction means in practice
- Why we clean the extracted text before saving or chunking it
- Why the ingestion flow should be separated from query-time logic

### Architecture in this stage

- Upload API receives a PDF
- PDF is saved to disk
- PyMuPDF extracts text page by page
- Text is cleaned and normalized
- The extracted content is returned for later chunking and embedding stages

### Why this matters

Without clean text, the downstream stages cannot work properly:

- chunking would split messy text
- embeddings would be built from poor quality inputs
- semantic retrieval would return low-quality results
- the LLM would get noisy context

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

## What the API returns

The endpoint returns:

- filename
- page count
- extracted page text

This is intentionally simple so the next stage can focus on chunking without the rest of the pipeline hiding the details.

## Next stage

After verifying PDF extraction, the next step is chunking with fixed-size chunks and overlap.
