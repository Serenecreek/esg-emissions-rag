from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path):

    # Load PDF
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    print(f"Loaded pages: {len(documents)}")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    print(f"Created chunks: {len(chunks)}")

    return chunks