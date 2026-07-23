# Contributing Guidelines

Thank you for your interest in contributing to **DocuMind AI – RAG Knowledge Retrieval Platform**!

## 1. Project Overview
DocuMind AI is a production-grade Retrieval-Augmented Generation (RAG) backend constructed with Python, FastAPI, LangChain, ChromaDB, HuggingFace Embeddings, Ollama, SQLite, and Docker.

---

## 2. Branch Naming Conventions
When introducing new features or bug fixes, create a feature branch off `main` using the following format:
- `feature/<short-description>` (e.g., `feature/hybrid-search`)
- `fix/<short-description>` (e.g., `fix/jwt-expiration-handling`)
- `docs/<short-description>` (e.g., `docs/readme-update`)

---

## 3. Commit Message Format
Follow standard semantic commit message guidelines:
- `feat: add hybrid keyword search service`
- `fix: handle empty vector database query exception`
- `docs: update API usage guidelines in README`
- `test: add unit test for query validation`

---

## 4. Code Style & Standards
- Follow **PEP 8** style guidelines for Python code.
- Include explicit type hints on function definitions.
- Maintain clear responsibility per service and router module.
- Format code using `black` or `flake8` standards.

---

## 5. Pull Request Process
1. Fork the repository and create your branch from `main`.
2. Ensure all tests pass by running:
   ```bash
   pytest tests/ -v
   ```
3. Submit a Pull Request detailing the changes, reasoning, and test verification results.
