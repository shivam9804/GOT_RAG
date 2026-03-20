# Project 2 - GOT RAG API (FastAPI + Ollama + Chroma)

This project is a FastAPI service that builds and queries a Retrieval-Augmented Generation (RAG) knowledge base for **"A Game of Thrones"** using:

- Parent-child chunking with LangChain `ParentDocumentRetriever`
- Chroma vector store for child embeddings
- Local parent doc store on disk
- Cross-encoder reranking (`BAAI/bge-reranker-base`)
- Ollama embeddings + chat model for final answers

## What It Does

- `GET /vector-db/`:
  - Loads `./app/books/1-A Game of Thrones.pdf`
  - Cleans text
  - Splits into parent and child chunks
  - Rebuilds local vector + parent stores from scratch

- `POST /query/`:
  - Retrieves parent documents via parent-child retriever
  - Reranks retrieved parent docs with a cross-encoder
  - Builds context from top results
  - Asks `llama3.2:3b` to answer strictly from retrieved context

## Project Structure

```text
.
├── main.py
├── app
│   ├── routers
│   │   ├── query_llm.py
│   │   └── vector_db.py
│   ├── schemas
│   │   └── query.py
│   └── services
│       ├── db_main.py
│       ├── query_llm.py
│       └── query_llm_parent_invoker.py
├── got_parent_child_chroma/   # generated at runtime
└── got_parent_store/          # generated at runtime
```

## Requirements

- Python **3.13+** (as declared in `pyproject.toml`)
- Ollama installed and running locally
- The book PDF available at:
  - `./app/books/1-A Game of Thrones.pdf`

## Install

Using `uv`:

```bash
uv sync
```

Or using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama Models Needed

Pull models used by this project:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Run The API

```bash
uv run fastapi dev main.py
```

Alternative:

```bash
uvicorn main:app --reload
```

Server default URL:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

## API Endpoints

### 1) Build / Rebuild Knowledge Base

`GET /vector-db/`

Example:

```bash
curl -X GET "http://127.0.0.1:8000/vector-db/"
```

Notes:

- This currently **deletes and recreates**:
  - `./got_parent_child_chroma`
  - `./got_parent_store`
- Run this before querying if your stores are missing or outdated.

### 2) Ask a Question

`POST /query/`

Request body:

```json
{
  "query": "Who finds the direwolf pups?"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"query":"Who finds the direwolf pups?"}'
```

Response shape:

```json
{
  "ranked_chunks": [
    {
      "reranker_score": 0.0,
      "metadata": {},
      "preview": "..."
    }
  ],
  "response": "Answer:\n...\n\nEvidence:\n..."
}
```

## Retrieval + Answering Pipeline

Implemented mainly in `app/services/query_llm_parent_invoker.py`:

1. Load Chroma child index (`got_parent_child`) with `nomic-embed-text`
2. Recreate parent/child splitters to match indexing settings
3. Load parent doc store from `./got_parent_store`
4. Retrieve parent docs using `ParentDocumentRetriever`
5. Rerank parent docs using `BAAI/bge-reranker-base`
6. Keep top 3 docs as context
7. Ask `llama3.2:3b` with a strict prompt requiring context-grounded answers

## Current Notes

- The service prints extensive retrieval/rerank debug logs to console.
- There is also a legacy `query_llm` implementation in `app/services/query_llm.py`, but the active `/query/` route uses `query_llm_parent`.
- If Hugging Face model download is needed for reranker on first run, internet access is required.

## Quick Start

1. Ensure PDF exists at `./app/books/1-A Game of Thrones.pdf`
2. Start Ollama
3. Pull required Ollama models
4. Start API
5. Call `GET /vector-db/`
6. Call `POST /query/`
