from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.schemas import ESGEmissionData


def create_rag_chain():

    # Create Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    # Force structured output
    structured_llm = llm.with_structured_output(
        ESGEmissionData
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are an ESG data extraction assistant.

        Extract greenhouse gas emissions from the
        provided sustainability report.

        Extract:

        - Reporting year
        - Scope 1 emissions
        - Scope 2 emissions
        - Scope 3 emissions
        - Unit for each emission scope

        Definitions:

        Scope 1:
        Direct greenhouse gas emissions from sources
        owned or controlled by the company.

        Scope 2:
        Indirect greenhouse gas emissions from purchased
        electricity, steam, heating or cooling.

        Scope 3:
        Other indirect emissions occurring across the
        company's value chain.

        IMPORTANT RULES:

        1. Use ONLY the provided context.

        2. Never invent emission values.

        3. If a value cannot be found, return null.

        4. Preserve the original unit.

        5. Do not confuse emission percentages with
           actual emission quantities.

        6. Do not confuse emission reduction values
           with absolute emissions.

        7. Prefer information from ESG tables when available.

        CONTEXT:
        {context}

        QUESTION:
        {question}
        """
    )

    chain = prompt | structured_llm

    return chain