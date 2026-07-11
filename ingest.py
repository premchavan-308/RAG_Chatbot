from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma


def create_vector_db(pdf_path):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embedding = MistralAIEmbeddings(
        model="mistral-embed"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="rag_pdf"
    )

    print("✅ Vector Database Created")

    return db
