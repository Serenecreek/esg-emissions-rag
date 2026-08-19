import json

from dotenv import load_dotenv

from src.ingestion import load_and_split_document
from src.retriever import (
    create_vector_store,
    create_retriever
)
from src.rag_chain import create_rag_chain
from src.utils import (
    build_context,
    get_source_pages
)


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# PDF path
# ============================================================

PDF_PATH = (
    r"C:\Users\52500\OneDrive - The Toro Company"
    r"\Downloads\edu\Janhvi\honda-SR-2026-en-all.pdf"
)


# ============================================================
# 1. Load and split PDF
# ============================================================

documents = load_and_split_document(
    PDF_PATH
)


# ============================================================
# 2. Create vector database
# ============================================================

vector_store = create_vector_store(
    documents
)


# ============================================================
# 3. Create retriever
# ============================================================

retriever = create_retriever(
    vector_store,
    k=5
)


# ============================================================
# 4. Create RAG chain
# ============================================================

rag_chain = create_rag_chain()


# ============================================================
# 5. Question
# ============================================================

question = """
What were the company's Scope 1, Scope 2,
and Scope 3 emissions?
"""


# ============================================================
# 6. Retrieve relevant documents
# ============================================================

retrieved_docs = retriever.invoke(
    question
)


# ============================================================
# 7. Build context
# ============================================================

context = build_context(
    retrieved_docs
)


# ============================================================
# 8. Generate structured answer
# ============================================================

result = rag_chain.invoke(
    {
        "context": context,
        "question": question
    }
)


# ============================================================
# 9. Convert Pydantic object to dictionary
# ============================================================

result_dict = result.model_dump()


# ============================================================
# 10. Display result
# ============================================================

print("\n")
print("=" * 70)
print("STRUCTURED ESG OUTPUT")
print("=" * 70)

print(
    json.dumps(
        result_dict,
        indent=4
    )
)


# ============================================================
# 11. Display sources
# ============================================================

source_pages = get_source_pages(
    retrieved_docs
)


print("\n")
print("=" * 70)
print("RETRIEVED SOURCE PAGES")
print("=" * 70)

print(source_pages)