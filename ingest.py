from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
import os
import shutil
import gc

def create_vector_db(pdf_path):

    gc.collect()

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db", ignore_errors=True)
    
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
    persist_directory="chroma_db",
    collection_name="rag_pdf"
    )

    print("✅ Vector Database Created")
