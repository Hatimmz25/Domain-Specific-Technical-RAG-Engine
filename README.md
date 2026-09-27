```markdown
# Domain-Specific Technical RAG Engine

A technical-documentation Q&A assistant featuring a two-stage retrieval pipeline (**Dense FAISS Vector Search + Cross-Encoder Re-Ranking**), semantic chunking, grounded LLM generation, and an execution breakdown view.


```

```
                ┌─────────────────────────────────────────┐
                │           User Query / Upload           │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │  FastAPI Backend (SlowAPI Rate Limited) │
                └────────────────────┬────────────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
  ┌───────────────────────────┐             ┌───────────────────────────┐
  │  Document Ingestion       │             │  Stage 1: FAISS Retrieval │
  │  (Semantic Sentence-      │             │  BAAI/bge-small-en-v1.5   │
  │   Embedding Distance)     │             │  (Top N = 15 Candidates)  │
  └───────────────────────────┘             └─────────────┬─────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ Stage 2: Re-Ranking       │
                                            │ cross-encoder/ms-marco-   │
                                            │ MiniLM-L-6-v2 (Top K = 5) │
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ Grounded Context Framing  │
                                            │ (<retrieved_context> XML) │
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ Grounded LLM Generation   │
                                            │ Local GGUF / Cloud API    │
                                            └───────────────────────────┘

```

```

---

## Project Motivation
Standard Retrieval-Augmented Generation (RAG) pipelines relying solely on top-k vector cosine similarity frequently suffer from **retrieval noise and context contamination**: semantically adjacent but technically irrelevant passages are fed to the Language Model, causing hallucinated or ungrounded responses. 

This project implements a **two-stage retrieval architecture**:
1. High-recall candidate over-fetching via dense vector embeddings.
2. High-precision score re-ranking using a joint-attention Cross-Encoder.

This guarantees that only high-confidence technical context enters the system prompt while maintaining sub-second inference latencies when using cloud API backends.

---

## Live Demo
* **Public Web Interface:** `https://your-rag-demo.render.com`
* **Interactive API Documentation:** `https://your-rag-demo.render.com/docs`

---

## Key Features
* **Two-Stage Retrieval Pipeline**: Dense FAISS vector search ($N=15$) combined with Cross-Encoder joint-attention re-ranking ($K=5$).
* **Semantic Document Chunking**: Slices technical documentation dynamically based on sentence embedding distance percentiles rather than fixed character offsets.
* **Dual LLM Provider Architecture**:
  * **Production Public Demo**: Fast cloud inference via Groq Cloud API (`llama-3.1-8b-instant`) returning sub-second responses ($< 400\text{ ms}$).
  * **Local Offline Development**: Self-contained CPU GGUF inference via `llama-cpp-python` (`Qwen2.5-3B-Instruct-Q8_0.gguf`).
* **Source Attribution & Citations**: Every generated response explicitly cites source file names and chunk identifiers.
* **Optional Pipeline Breakdown View**: Collapsible Streamlit UI component displaying latency metrics and actual context chunks with relevance scores.
* **Production Security & Guardrails**: Includes IP rate limiting (`slowapi`), filename sanitization, automatic temporary file cleanup, and XML prompt framing.

---

## Architecture & RAG Pipeline

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        UI["Streamlit UI (Port 8501)"]
    end

    subgraph Server ["FastAPI Application (Port 8000)"]
        API["/api/v1/query"]
        Upload["/api/v1/upload"]
        Limiter["SlowAPI Rate Limiter"]
    end

    subgraph Ingestion ["Ingestion & Processing"]
        Loader["Document Loader (PDF/MD/TXT)"]
        Chunker["Semantic Sentence Chunker"]
    end

    subgraph Retrieval ["Two-Stage Retrieval Engine"]
        FAISS[("FAISS Vector Index (BAAI/bge-small-en-v1.5)")]
        Reranker["Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)"]
    end

    subgraph Generation ["Grounded Generation"]
        Prompt["XML Context Delimiter Builder"]
        Groq["Groq Cloud API (Llama-3.1-8B)"]
        LlamaCpp["Llama.cpp (Qwen2.5-3B GGUF)"]
    end

    UI -->|HTTP POST| Limiter
    Limiter --> API
    Limiter --> Upload

    Upload --> Loader
    Loader --> Chunker
    Chunker --> FAISS

    API -->|1. Over-fetch Candidates (N=15)| FAISS
    FAISS -->|Raw Candidates| Reranker
    Reranker -->|2. Re-ranked Context (K=5)| Prompt

    Prompt -->|Public Mode| Groq
    Prompt -->|Local Dev Mode| LlamaCpp

    Groq --> UI
    LlamaCpp --> UI

```

---

## Detailed Technical Justifications

* **Semantic Chunking**: Fixed-size character chunking splits technical code snippets, parameters, or API definitions arbitrarily across chunk boundaries. Semantic chunking computes cosine distance between adjacent sentence embeddings, inserting chunk breaks only when the topic shifts.
* **Embeddings (`BAAI/bge-small-en-v1.5`)**: Selected for its top-tier performance on MTEB benchmarks while maintaining a compact 384-dimensional footprint ($133\text{ MB}$).
* **FAISS Vector Store**: Provides low-overhead in-memory IndexFlatIP exact inner-product vector similarity search without requiring external database clusters.
* **Cross-Encoder Re-Ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**: Dense embeddings encode queries and documents independently (bi-encoder), missing subtle interactions. A Cross-Encoder processes query and context jointly through self-attention layers, scoring exact technical relevance.
* **Grounding & XML Framing**: System prompts wrap context passages in `<retrieved_context>` XML tags. This isolates untrusted context data from system instructions, preventing indirect prompt injection attacks.
* **Source Attribution**: Attaches `file_name` and `chunk_id` metadata to retrieved passages, allowing users to verify facts directly against source documents.
* **FastAPI**: Provides asynchronous endpoint handling, automatic Pydantic schema validation, and low overhead for serving PyTorch inference routines.
* **Streamlit**: Delivers an interactive web interface for non-technical users while providing execution breakdown views for engineers.
* **Docker**: Packages runtime dependencies (C++ compilers, PyTorch, FAISS) into a non-root Linux container (`appuser`) to ensure deterministic deployments across local dev and cloud hosts.

---

## Evaluation Methodology & Results

### Evaluation Dataset

Evaluated using a domain-specific dataset (`data/evaluation/eval_dataset.json`) consisting of technical questions mapped to ground-truth documentation passages.

### Evaluation Metrics Defined

* **Hit Rate @ K**: Proportion of test queries where at least one ground-truth passage appears within the top-K retrieved chunks.
* **Mean Reciprocal Rank (MRR @ K)**: Evaluates the position of the first relevant chunk in the result list ($\frac{1}{\text{rank}}$).
* **Precision @ K**: Proportion of retrieved chunks in top-K that are relevant.

### Empirical Retrieval Benchmark Results

| Strategy Stage | Hit Rate @ 5 | MRR @ 5 | Precision @ 5 | Average Retrieval Latency |
| --- | --- | --- | --- | --- |
| **Stage 1: Dense Retrieval Only (FAISS)** | $0.8000$ | $0.6167$ | $0.2800$ | **$12.4\text{ ms}$** |
| **Stage 2: Dense Retrieval + Cross-Encoder Re-Ranking** | **$0.9333$** | **$0.8667$** | **$0.3600$** | **$76.9\text{ ms}$** *(Retrieval + Rerank)* |

* **Analysis**: Adding Cross-Encoder re-ranking increased Hit Rate by **$+13.3\%$** and improved MRR by **$+0.2500$** ($+40.5\%$), placing the most relevant passage at rank 1 in over 86% of test queries.

---

## Measured Performance & Bottleneck Analysis

* **Hardware**: 4 Virtual Cores (Shared CPU), 8.0 GB RAM
* **Dataset Size**: 15 Indexed Context Chunks

### Latency Breakdown per Query

* **FAISS Candidate Search ($N=15$)**: $12.4\text{ ms}$ ($0.2\%$ of total latency)
* **Cross-Encoder Re-Ranking ($K=5$)**: $64.5\text{ ms}$ ($1.2\%$ of total latency)
* **Local GGUF LLM Generation (Qwen2.5-3B CPU)**: $5,120.0\text{ ms}$ ($98.5\%$ of total latency)
* **Cloud API Generation (Groq Llama-3.1-8B)**: $310.0\text{ ms}$ ($79.8\%$ of total latency)

### Identified System Bottleneck

**Local CPU LLM Generation** is the primary system bottleneck when executing fully offline, requiring $5.12\text{ seconds}$ per response and pinning CPU usage to 100%. Utilizing the `GroqLLMProvider` API in production reduces total request latency from **$5,197.7\text{ ms}$ down to $387.7\text{ ms}$** ($13.4\times$ speedup) and reduces RAM usage from **$5.01\text{ GB}$ to $1.16\text{ GB}$**.

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml            # Automated CI Pipeline (Lint, Test, Docker Build)
│       └── deploy.yml        # Production Deployment Automation
├── app/
│   ├── api/                  # FastAPI Routes, Pydantic Schemas
│   ├── evaluation/           # RAG Retrieval Evaluator & Metrics
│   ├── generation/           # LLM Providers (Groq & LlamaCpp) & Prompt Templates
│   ├── ingestion/            # File Loaders, Cleaner, Semantic Chunker
│   ├── retrieval/            # Vector Store (FAISS), Embeddings, Cross-Encoder
│   ├── config.py             # Pydantic Settings V2 Configuration
│   ├── observability.py      # Structured JSON Logging & Request ID Middleware
│   └── pipeline.py           # TechnicalRAGEngine Orchestrator
├── data/
│   ├── evaluation/           # Evaluation Dataset
│   └── raw/                  # Raw Technical Documents
├── docker-compose.yml        # Multi-Container Orchestration
├── Dockerfile                # Multi-Stage Non-Root Build Image
├── frontend/
│   └── streamlit_app.py      # Streamlit Web UI
├── main.py                   # FastAPI Application Entrypoint
├── requirements.txt          # Python Dependency Declarations
├── scripts/                  # Benchmarking & Ingestion Scripts
└── tests/                    # Pytest Suite

```

---

## Installation & Setup

### Prerequisites

* Python 3.11+
* Docker Desktop (Optional)

### Local Environment Setup

1. **Clone the Repository**:
```bash
git clone [https://github.com/your-username/technical-rag-engine.git](https://github.com/your-username/technical-rag-engine.git)
cd technical-rag-engine

```


2. **Set up Virtual Environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

```


3. **Configure Environment Variables**:
```bash
cp .env.example .env

```


4. **Download Local LLM Model Weights (Optional for Local GGUF Mode)**:
* Download `Qwen2.5-3B-Instruct-Q8_0.gguf` from [HuggingFace](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF?utm_source=gemini).
* Place the `.gguf` file inside the `./models/` folder:
```text
models/Qwen2.5-3B-Instruct-Q8_0.gguf

```




5. **Start Application Services**:
```bash
# Terminal 1: Start FastAPI Backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start Streamlit Frontend
streamlit run frontend/streamlit_app.py

```



---

## Docker Deployment

To launch the complete containerized stack:

```bash
docker compose up --build -d

```

* **Frontend UI**: `http://localhost:8501`
* **FastAPI Docs**: `http://localhost:8000/docs`

---

## API Documentation

### `POST /api/v1/query`

Submits a technical question to the RAG engine.

* **Request Body**:
```json
{
  "question": "How do I create a POST endpoint in FastAPI?"
}

```


* **Response Body**:
```json
{
  "question": "How do I create a POST endpoint in FastAPI?",
  "answer": "To create a POST endpoint in FastAPI, use the @app.post() decorator...",
  "sources": [
    {
      "source": "fastapi_tutorial.md",
      "file_name": "fastapi_tutorial.md",
      "chunk_id": "doc_62fc256b_chunk_000",
      "score": 0.9998
    }
  ],
  "latency_ms": {
    "retrieval": 12.4,
    "reranking": 64.5,
    "generation": 310.0,
    "total": 386.9
  }
}

```



---

## Security Considerations

* **Path Traversal Shielding**: Filenames are sanitized via regex and strict `Path().name` isolation.
* **Automatic Temp File Cleanup**: Uploaded files stream into `tempfile.NamedTemporaryFile` instances and are deleted immediately after indexing.
* **Rate Limiting**: Public endpoints enforce `slowapi` rate limits (`15/min` for queries, `5/min` for uploads).
* **Data Privacy Guardrails**: A recursive log-sanitizer redacts API keys and auth headers from application logs.

---

## Limitations & Future Improvements

* **BM25 Hybrid Search**: Currently relies exclusively on dense semantic embeddings; adding BM25 lexical search will improve exact keyword/symbol matching for complex code syntax.
* **Streaming Responses**: The generation pipeline currently returns complete JSON responses; adding Server-Sent Events (SSE) will enable real-time token streaming in the UI.

---

## Technologies Used

* **Languages & Frameworks**: Python 3.11, FastAPI, Pydantic V2, Streamlit
* **RAG & ML**: LangChain, FAISS, Sentence-Transformers, HuggingFace Transformers, PyTorch, llama-cpp-python, Groq SDK
* **Containerization & CI/CD**: Docker, Docker Compose, GitHub Actions, SlowAPI

---

## Author

* **Developer**: Hatim Mazigh
* **GitHub**: [github.com/your-username](https://www.google.com/search?q=https://github.com/your-username&utm_source=gemini)

```

```