def build_context(documents):

    context_parts = []

    for doc in documents:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        # Convert zero-based PDF page number
        # to human-readable page number
        if isinstance(page, int):
            page = page + 1

        context_parts.append(
            f"""
SOURCE PAGE: {page}

{doc.page_content}
"""
        )

    return "\n\n".join(
        context_parts
    )


def get_source_pages(documents):

    pages = []

    for doc in documents:

        page = doc.metadata.get(
            "page"
        )

        if isinstance(page, int):
            page = page + 1

        if page is not None:
            pages.append(page)

    return pages