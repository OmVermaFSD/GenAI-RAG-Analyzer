# DocuMind AI - Production-Grade RAG Knowledge Retrieval Backend

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.1-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.11-1C3C3C.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama--3-black.svg)](https://ollama.com/)

**DocuMind AI** is an enterprise-grade Retrieval-Augmented Generation (RAG) backend designed for real-time PDF document ingestion, vector similarity search, and local LLM-powered context synthesis. Built using **Python**, **FastAPI**, **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Ollama (Llama 3)**.

---

## 🌟 Key System Capabilities

- **PDF Ingestion Pipeline**: Asynchronous multi-page PDF processing using `PyPDFLoader`.
- **Recursive Text Chunking**: Intelligent document chunking using `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) to preserve context boundaries.
- **HuggingFace Vector Embeddings**: High-performance local embeddings powered by `sentence-transformers/all-MiniLM-L6-v2`.
- **Persistent Vector Database**: ChromaDB store with HNSW Cosine Similarity indexing stored locally at `./chroma_db_store`.
- **Local Privacy-First LLM Synthesis**: Fully offline context-aware question answering with Ollama (`llama3`).
- **Structured JSON Responses**: Strict Pydantic v2 validated request and response payloads, exposing detailed source document citations.

---

## 🏗️ System Architecture

```
                                  +---------------------------------------+
                                  |            Client / Front-End         |
                                  +-------------------+-------------------+
                                                      |
                                           HTTP POST /ingest (PDF)
                                           HTTP POST /query  (JSON)
                                                      v
                                  +---------------------------------------+
                                  |         FastAPI Web Service           |
                                  |       (app.py - Async Engine)         |
                                  +---------+-------------------+---------+
                                            |                   |
                     PDF Document Ingestion |                   | RAG Query Pipeline
                                            v                   v
+-------------------------------------------------+     +-------------------------------------------------+
| PyPDF Extraction & Recursive Splitter           |     | Vector Cosine Similarity Search                 |
| (chunk_size=1000, chunk_overlap=200)            |     | (Top-K context retrieval)                       |
+-----------------------+-------------------------+     +-----------------------+-------------------------+
                        |                                                       |
                        v                                                       v
+-------------------------------------------------+     +-------------------------------------------------+
| HuggingFace Embedding Generation                |     | Ollama Local LLM Synthesis                      |
| (sentence-transformers/all-MiniLM-L6-v2)        |     | (Llama 3 model)                                 |
+-----------------------+-------------------------+     +-----------------------+-------------------------+
                        |                                                       |
                        v                                                       v
+-------------------------------------------------+     +-------------------------------------------------+
| ChromaDB Vector Database                        |     | Structured JSON Output                          |
| (Persistent Storage: ./chroma_db_store)         |====>| (Answer + Source Document Context Chunks)       |
+-------------------------------------------------+     +-------------------------------------------------+
```

---

## 🛠️ Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Framework** | [FastAPI](https://fastapi.tiangolo.com/) | Async Web Framework with Pydantic v2 Validation |
| **Orchestration** | [LangChain](https://www.langchain.com/) | RAG Chain, Document Loaders, Splitters, and Retrievers |
| **Document Parser** | [PyPDF](https://pypdf.readthedocs.io/) | High-speed PDF text & page extraction |
| **Embeddings** | [HuggingFace](https://huggingface.co/) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Local persistent vector database (`./chroma_db_store`) |
| **Local LLM** | [Ollama](https://ollama.com/) | Llama 3 local LLM synthesis engine |
| **Server** | [Uvicorn](https://www.uvicorn.org/) | High-performance ASGI HTTP Server |

---

## 🚀 Quickstart & Local Setup Guide

### 1. Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Installed and running locally ([Download Ollama](https://ollama.com/download))

Pull the Llama 3 model in terminal:
```bash
ollama pull llama3
```

Ensure the Ollama server is active at `http://localhost:11434`.

---

### 2. Installation

Clone the repository and create a Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

Install pinned dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Launching the Service

Start the FastAPI application via Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The service will be accessible at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 📡 API Endpoints Documentation

### 1. Health Check
`GET /health`

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "vector_store_initialized": true,
  "total_documents": 42
}
```

---

### 2. Ingest Document
`POST /ingest`

Uploads a PDF file, extracts content, splits text into chunks, computes embeddings, and stores them in ChromaDB.

**Request**: `multipart/form-data` with key `file` (PDF file).

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

**Response (`201 Created`)**:
```json
{
  "filename": "document.pdf",
  "status": "success",
  "total_chunks": 18,
  "message": "Successfully ingested and indexed 18 vector chunks into ChromaDB."
}
```

---

### 3. Query Knowledge Base
`POST /query`

Executes cosine similarity retrieval and generates an answer using Ollama Llama 3.

**Request Body (`application/json`)**:
```json
{
  "question": "What are the core technical requirements discussed in the document?"
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the core technical requirements discussed in the document?"}'
```

**Response (`200 OK`)**:
```json
{
  "query": "What are the core technical requirements discussed in the document?",
  "answer": "The core technical requirements include implementing a single-app RAG architecture using FastAPI, LangChain, ChromaDB, HuggingFace embeddings, and Ollama Llama 3...",
  "source_documents": [
    {
      "page_content": "Technical Requirements: 1. Python 3.10+ with FastAPI async endpoints...",
      "metadata": {
        "page": 0,
        "source_filename": "document.pdf"
      }
    }
  ]
}
```

---

## 🧪 Testing & Verification

1. Start Uvicorn: `uvicorn app:app --port 8000`
2. Open Swagger UI at `http://localhost:8000/docs`.
3. Use the `/ingest` endpoint to upload sample PDF files.
4. Verify that ChromaDB files are generated under `./chroma_db_store`.
5. Send natural language questions to `/query` and review the returned answer along with source document page metadata.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
