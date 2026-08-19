from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings


def create_vector_store(documents):

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


def create_retriever(
    vector_store,
    k=5
):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )

    return retriever