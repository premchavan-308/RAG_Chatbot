from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)

from langchain_core.prompts import ChatPromptTemplate

embedding = MistralAIEmbeddings(
    model="mistral-embed"
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

template = """
You are an AI assistant that answers ONLY from the uploaded PDF.

Rules:
- Use only the provided context.
- Never use outside knowledge.
- If the answer is missing from the context, reply:
"I couldn't find this information in the uploaded PDF."
- Mention page numbers whenever possible.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


def ask(question):

    db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
    collection_name="rag_pdf"
    )

    retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
    )


    docs = retriever.invoke(question)

    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}")
        print(doc.page_content[:300])
        print("-" * 50)

    context = ""

    for doc in docs:

        page = doc.metadata.get("page", 0)

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
