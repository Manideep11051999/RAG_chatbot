# AI Customer Service Chatbot (RAG Prototype)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot for a large automotive company.

The chatbot answers customer questions using internal knowledge base documents.  
It runs locally using Ollama and demonstrates a full end-to-end RAG pipeline.

---

## Business Context

Automotive companies receive repetitive customer inquiries about:
- Vehicle features
- Service schedules
- Warranty coverage
- Ordering process

Instead of manually answering, this chatbot:
- Retrieves relevant internal documents
- Generates grounded responses
- Displays sources for transparency

This improves customer experience and operational efficiency.

---

## Architecture Overview

1. Documents are loaded from `/data/knowledge_base`
2. Documents are chunked using RecursiveCharacterTextSplitter
3. Chunks are embedded using Ollama Embeddings (`nomic-embed-text`)
4. Stored in ChromaDB vector database
5. User query is embedded
6. Top-K similar chunks are retrieved
7. Context is injected into a strict system prompt
8. Local LLM (Llama 3.2) generates answer
9. Sources are displayed in UI
10. The system ensures that the LLM answers are grounded strictly in retrieved context to minimize hallucinations.

---

## Tech Stack

- Ollama (Local LLM + Embeddings)
- Llama 3.2 (3B)
- ChromaDB (Vector Store)
- LangChain (RAG framework)
- Streamlit (Chat Interface)

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull required models
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Run the application
```bash
streamlit run src/app.py
```

---

## Key Technical Decisions

- Used RAG instead of fine-tuning for flexibility
- Used local models due to CPU constraint requirements
- Top-K retrieval set to 3 for balanced recall vs latency
- A lightweight 3B parameter model was selected to balance reasoning quality and local CPU performance.
- Strict system prompt to reduce hallucinations

---

## Limitations

- Runs locally (CPU bound, slower inference)
- Small model may miss complex reasoning
- Basic similarity search (no hybrid retrieval yet)

---

## Future Improvements

- Hybrid search (BM25 + embeddings)
- Reranking layer
- API backend deployment
- Production-ready architecture with caching
- Better evaluation metrics

---

## Presentation

The project presentation slides are included in this repository.

---

## Author

Manideep  
AI Engineer Intern Case Study Submission
