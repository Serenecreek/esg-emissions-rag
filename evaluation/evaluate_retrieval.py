from dotenv import load_dotenv

from src.ingestion import load_and_split_document
from src.retriever import (
    create_vector_store,
    create_retriever
)

from evaluation.evaluation_dataset import (
    evaluation_dataset
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# PDF
# ============================================================

PDF_PATH = (
    r"C:\Users\52500\OneDrive - The Toro Company"
    r"\Downloads\edu\Janhvi\honda-SR-2026-en-all.pdf"
)


# ============================================================
# Metrics
# ============================================================

def precision_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    retrieved = retrieved_pages[:k]

    relevant_retrieved = (
        set(retrieved)
        &
        set(relevant_pages)
    )

    return (
        len(relevant_retrieved)
        /
        k
    )


def recall_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    retrieved = retrieved_pages[:k]

    relevant_retrieved = (
        set(retrieved)
        &
        set(relevant_pages)
    )

    if not relevant_pages:
        return 0

    return (
        len(relevant_retrieved)
        /
        len(relevant_pages)
    )


def hit_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    retrieved = set(
        retrieved_pages[:k]
    )

    relevant = set(
        relevant_pages
    )

    return int(
        bool(
            retrieved & relevant
        )
    )


# ============================================================
# Build vector database
# ============================================================

documents = load_and_split_document(
    PDF_PATH
)

vector_store = create_vector_store(
    documents
)

retriever = create_retriever(
    vector_store,
    k=5
)


# ============================================================
# Run evaluation
# ============================================================

results = []


for item in evaluation_dataset:

    question = item[
        "question"
    ]

    relevant_pages = item[
        "relevant_pages"
    ]


    # Retrieve
    docs = retriever.invoke(
        question
    )


    # Convert PDF pages to human pages
    retrieved_pages = []


    for doc in docs:

        page = doc.metadata.get(
            "page"
        )

        if isinstance(page, int):

            page = page + 1

        if page is not None:

            retrieved_pages.append(
                page
            )


    # Metrics
    precision = precision_at_k(
        retrieved_pages,
        relevant_pages,
        5
    )


    recall = recall_at_k(
        retrieved_pages,
        relevant_pages,
        5
    )


    hit = hit_at_k(
        retrieved_pages,
        relevant_pages,
        5
    )


    results.append(
        {
            "question": question,
            "retrieved_pages": retrieved_pages,
            "relevant_pages": relevant_pages,
            "precision": precision,
            "recall": recall,
            "hit": hit
        }
    )


# ============================================================
# Print individual results
# ============================================================

print()
print("=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)


for result in results:

    print()
    print(
        "Question:",
        result["question"]
    )

    print(
        "Retrieved:",
        result["retrieved_pages"]
    )

    print(
        "Relevant:",
        result["relevant_pages"]
    )

    print(
        f"Precision@5: "
        f"{result['precision']:.3f}"
    )

    print(
        f"Recall@5: "
        f"{result['recall']:.3f}"
    )

    print(
        f"Hit@5: "
        f"{result['hit']}"
    )


# ============================================================
# Average metrics
# ============================================================

average_precision = sum(
    result["precision"]
    for result in results
) / len(results)


average_recall = sum(
    result["recall"]
    for result in results
) / len(results)


average_hit = sum(
    result["hit"]
    for result in results
) / len(results)


# ============================================================
# Final result
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Average Precision@5: "
    f"{average_precision:.3f}"
)

print(
    f"Average Recall@5: "
    f"{average_recall:.3f}"
)

print(
    f"Average Hit@5: "
    f"{average_hit:.3f}"
)