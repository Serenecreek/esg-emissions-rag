import tempfile

import streamlit as st

from dotenv import load_dotenv

from src.ingestion import load_and_split_document
from src.retriever import (
    create_vector_store
)
from src.rag_chain import create_rag_chain
from src.utils import (
    build_context,
    get_source_pages
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()


st.set_page_config(
    page_title="ESG Emissions RAG",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# Header
# ============================================================

st.title(
    "ESG Emissions Intelligence Assistant"
)

st.write(
    """
    Extract Scope 1, Scope 2 and Scope 3 emissions
    from sustainability reports using RAG.
    """
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "RAG Configuration"
)


top_k = st.sidebar.slider(
    "Retrieved documents",
    min_value=1,
    max_value=10,
    value=5
)


# ============================================================
# Upload PDF
# ============================================================

uploaded_file = st.file_uploader(
    "Upload ESG / Sustainability Report",
    type=["pdf"]
)


if uploaded_file is not None:

    # ========================================================
    # Save uploaded PDF temporarily
    # ========================================================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        pdf_path = temp_file.name


    # ========================================================
    # Process document
    # ========================================================

    with st.spinner(
        "Processing sustainability report..."
    ):

        documents = load_and_split_document(
            pdf_path
        )


    st.success(
        f"Processed {len(documents)} chunks."
    )


    # ========================================================
    # Create vector store
    # ========================================================

    with st.spinner(
        "Creating vector database..."
    ):

        vector_store = create_vector_store(
            documents
        )


    # ========================================================
    # Create retriever
    # ========================================================

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k
        }
    )


    # ========================================================
    # Create RAG chain
    # ========================================================

    rag_chain = create_rag_chain()


    # ========================================================
    # Question
    # ========================================================

    st.subheader(
        "Ask a question"
    )


    question = st.text_input(
        "Your question",
        placeholder=(
            "What were the Scope 1, Scope 2 "
            "and Scope 3 emissions?"
        )
    )


    # ========================================================
    # Run pipeline
    # ========================================================

    if st.button(
        "Extract ESG Data"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving and extracting ESG data..."
            ):

                # Retrieve
                retrieved_docs = retriever.invoke(
                    question
                )

                # Build context
                context = build_context(
                    retrieved_docs
                )

                # LLM extraction
                result = rag_chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )


            # =================================================
            # Structured output
            # =================================================

            st.subheader(
                "Structured ESG Data"
            )

            result_dict = result.model_dump()

            st.json(
                result_dict
            )


            # =================================================
            # Reporting year
            # =================================================

            st.subheader(
                "Reporting Year"
            )

            st.write(
                result.reporting_year
                if result.reporting_year
                else "Not found"
            )


            # =================================================
            # Emission metrics
            # =================================================

            st.subheader(
                "Emissions"
            )

            col1, col2, col3 = st.columns(3)


            with col1:

                value = result.scope_1.value
                unit = result.scope_1.unit

                st.metric(
                    "Scope 1",
                    (
                        f"{value:,.0f} {unit}"
                        if value is not None
                        else "Not found"
                    )
                )


            with col2:

                value = result.scope_2.value
                unit = result.scope_2.unit

                st.metric(
                    "Scope 2",
                    (
                        f"{value:,.0f} {unit}"
                        if value is not None
                        else "Not found"
                    )
                )


            with col3:

                value = result.scope_3.value
                unit = result.scope_3.unit

                st.metric(
                    "Scope 3",
                    (
                        f"{value:,.0f} {unit}"
                        if value is not None
                        else "Not found"
                    )
                )


            # =================================================
            # Source pages
            # =================================================

            st.subheader(
                "Retrieved Sources"
            )

            source_pages = get_source_pages(
                retrieved_docs
            )

            st.write(
                f"Source pages: {source_pages}"
            )


            # =================================================
            # Source chunks
            # =================================================

            for index, doc in enumerate(
                retrieved_docs,
                start=1
            ):

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                if isinstance(page, int):
                    page = page + 1


                with st.expander(
                    f"Source {index} | Page {page}"
                ):

                    st.write(
                        doc.page_content
                    )