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
You are a helpful AI assistant.

Answer ONLY from the supplied context.

If the answer is not available in the context,
say "I don't know."

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
        "k": 3,
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

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return result.content