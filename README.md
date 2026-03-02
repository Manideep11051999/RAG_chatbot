# Case Study: AI Engineer Intern – Customer Journey Analytics & Data Science

---

## Overview

| Topic | Details |
|---|---|
| **Position** | Intern AI Engineer – Customer Journey Analytics & Data Science |
| **Deliverables** | Part A: GitHub Repository (Code) · Part B: Presentation (max. 20 minutes) |
| **Language** | English |
| **Start Date** | Friday, 27.02.2026 |
| **Deadline** | Tuesday, 03.03.2026, 23:59 — all commits pushed to a **public** GitHub repository |

---

## Context & Scenario

You are joining the **Customer Journey Analytics & Data Science** team at a large automotive company. The team is responsible for data-driven analysis and optimization of digital customer channels – including websites, vehicle configurators, and service portals.

Every day, the company receives hundreds of customer inquiries. Many relate to recurring topics: vehicle features, service schedules, warranty terms, or ordering processes. This information already exists in internal knowledge bases but is not always easy to find.

**Your task:** Build a prototype of a **local AI-powered chatbot** that answers customer questions based on a provided knowledge base. The chatbot should use **Retrieval-Augmented Generation (RAG)** to find relevant documents and generate natural language answers.

> **Important:** The focus is on **technical implementation, architecture, and your decision-making process**, not on answer quality. Since the prototype runs locally, we understand there are CPU/GPU constraints. Lightweight models are perfectly fine. We want to see that you understand the concepts and can explain **why** you chose what you chose.

---

## Part A: Technical Implementation

### A1 – Setup & Knowledge Base

1. **Set up a local LLM** using [Ollama](https://ollama.ai):
   - A **Chat model** — a lightweight model works
   - An **Embedding model**
2. Use the provided documents in `data/knowledge_base/`
3. Implement a **Document Ingestion Pipeline**:
   - Load and process documents
   - Generate embeddings using your chosen Ollama embedding model
   - Store embeddings in a **Vector Store** — we recommend [ChromaDB](https://docs.trychroma.com/) for simplicity, but you may use alternatives

### A2 – Implement RAG Pipeline

Build the core chatbot logic:

1. **Retrieval:** For a user query, retrieve the most relevant chunks via similarity search
2. **Augmentation:** Inject the retrieved context into a prompt
3. **Generation:** The chat model generates an answer based on the context

**Requirements:**
- Number of retrieved chunks should be configurable (Top-K)
- System prompt should instruct the model to answer based on provided context
- Sources (document titles) should be referenced in the answer

**Framework:** Use **LangChain / LangGraph** as your framework.


### A3 – Chat Interface

Build a simple chat interface (e.g. using **Streamlit**):

- Input field for queries
- Display of chatbot response
- Display of source documents used
- Chat history within a session

### A4 – Documentation

Make sure to properly document your code.

---

## Part B: Presentation

Create a **presentation (max. 20 minutes)** for a **mixed audience** — both business stakeholders and technical team members should be able to follow along.

The goal is to show that you understand not just **how** to build this, but **why** it matters and where it could go from here.

### What we'd like to see:

- **Business perspective:** What problem does an AI chatbot solve? Why is this relevant for an automotive company? What value does it create?
- **Solution & Architecture:** High-level overview of your RAG architecture. Why RAG (vs. other approaches)? Explain your key technical decisions and trade-offs.
- **Demo & Results:** Show your prototype in action (screenshots or preferably **live demo**). What works well, what are the limitations?
- **Roadmap:** Where could this go next? How would you scale this to production? What would change (models, infrastructure, integrations)? What further use cases do you see?
- **Summary & Q&A**

### Evaluation Criteria

| Criterion | What we look for |
|---|---|
| **Reasoning & Decision-Making** | Clear rationale for technical choices (model, chunk size, retrieval strategy, etc.). We want to understand *why*, not just *what*. |
| **Business Perspective** | Ability to frame the technical solution in business terms — impact on customer experience, operational efficiency, and strategic value. |
| **Technical Understanding** | Solid grasp of RAG concepts, LLM fundamentals, and the end-to-end pipeline. |
| **Communication** | Presenting to a mixed audience — making it accessible for business stakeholders while staying precise for engineers. |

---

## How to Submit

1. Click **"Use this template"** → **"Create a new repository"** on this GitHub page
2. Create your own **public** repository from this template
3. Work in your own repo — commit and push your code as you go (Make sure to also push the ppt!)
4. When you're done, share the **link to your repository** with us
5. **All commits must be pushed before the deadline** — late submissions will not be considered

---

## Tips

- **Prioritize a working end-to-end pipeline** over perfection in any single component.
- **Use small models** (1B–4B params). Answer quality is secondary. We evaluate your implementation and understanding.
- **Explain your decisions** — we want to understand your thought process.
- A basic **Streamlit app template** is provided in `src/app.py` — you can use it as a starting point or build your own from scratch.

**Good luck!**
