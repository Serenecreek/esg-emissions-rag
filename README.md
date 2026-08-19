# ESG Emissions RAG

## Overview

An AI-powered ESG document intelligence system that uses
Retrieval-Augmented Generation to extract Scope 1, Scope 2
and Scope 3 greenhouse gas emissions from sustainability reports.

## Features

- PDF document ingestion
- Semantic document chunking
- Hugging Face embeddings
- FAISS vector search
- LangChain RAG pipeline
- Structured Pydantic output
- Scope 1/2/3 extraction
- Source-page traceability
- Streamlit interface
- Retrieval evaluation

## Architecture

PDF
→ Chunking
→ Embeddings
→ FAISS
→ Retriever
→ LLM
→ Structured Output
→ ESG JSON

## Tech Stack

Python
LangChain
FAISS
Hugging Face
Groq
Pydantic
Streamlit

## Evaluation

Precision@K
Recall@K
Hit@K
Extraction Accuracy

## How to Run

1. Clone repository
2. Create virtual environment
3. Install requirements
4. Add GROQ_API_KEY to .env
5. Add ESG PDF
6. Run Streamlit

streamlit run app.py
