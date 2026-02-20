
# AI Research Assistant

An AI research assistant designed to streamline academic workflows including paper discovery, document-grounded Q&A, structured reporting, PDF summarization, and code generation.

---

## 🚀 Overview

This system integrates LLM reasoning, multi-source academic search  into a clean architecture using Groq-hosted LLaMA models.

The assistant prioritizes:

* Context-grounded answers
* Structured academic outputs
* Clean modular design

---

## 🧠 Core Capabilities

### 🔎 Multi-Source Paper Search

* Semantic Scholar API
* arXiv API
* Result deduplication & ranking
* Citation-aware sorting

### 📄 Paper Report Generation

Generates structured academic reports with:

1. Executive Summary
2. Key Contributions
3. Methodology
4. Strengths
5. Limitations
6. Applications
7. Future Work

Strict no-hallucination rules.

---

### 📚 PDF-Based Q&A

* Extracts and cleans PDF text using PyPDF2
* Chunk-based segmentation
* Keyword scoring for relevant chunk retrieval
* Context-grounded answers only

If answer is not found → explicitly states it.

---

### ⚖️ Paper Comparison

Compares two papers across:

* Similarities
* Differences
* Strengths
* Final Verdict

---

### 🧑‍💻 Code Generation

* Production-grade code generation
* Clean structure
* Error handling
* Language configurable

---



## 🛠️ Tech Stack

* Python
* Groq API (LLaMA 3.x models)
* AutoGen
* Semantic Scholar API
* arXiv API
* PyPDF2
* dotenv
* concurrent.futures

---

## 🔐 Environment Variables

Create a `.env` file:

GROQ_API_KEY=your_key_here

---

## ⚙️ Design Philosophy

* Accuracy-first prompting
* No fabricated academic claims
* Clean separation of responsibilities
* Deterministic temperature tuning for research tasks

---

## 📌 Future Improvements

* Vector database integration (FAISS / Chroma)
* Persistent chat sessions
* Web UI (Streamlit or FastAPI frontend)
* Rate limiting & caching layer
* Logging & evaluation metrics

---

## 📄 License

No license specified yet.

---


