from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_mistralai import (
    ChatMistralAI,
)

from langchain_core.prompts import ChatPromptTemplate


llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

template = """
You are an intelligent RAG assistant.

Answer ONLY from the provided context.

Rules:

- Never use outside knowledge.
- If the answer isn't present in the context,
reply:
"I couldn't find this information in the uploaded PDF."

- Mention page numbers whenever possible.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


def ask(question):

    db = st.session_state.db

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(question)

    if len(docs) == 0:
        return "I couldn't find anything in the uploaded PDF."

    print("\nRetrieved Chunks\n")

    context = ""

    for i, doc in enumerate(docs):

        page = doc.metadata.get("page", "Unknown")

        print(f"\nChunk {i+1}")
        print(f"Page : {page+1}")
        print(doc.page_content[:300])

        context += f"\n(Page {page+1})\n"
        context += doc.page_content
        context += "\n\n"

    chain = prompt | llm

    result = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return result.content
