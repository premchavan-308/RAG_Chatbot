import streamlit as st
import tempfile

from rag import ask
from ingest import create_vector_db

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Chatbot")

st.write("Upload any PDF and ask questions from it.")

# ---------------- Upload PDF ---------------- #

uploaded_file = st.file_uploader(
    "📄 Drag and Drop your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF Uploaded")

    if st.button("Create Vector Database"):

        with st.spinner("Creating Vector Database..."):

            create_vector_db(pdf_path)

        st.success("✅ Vector Database Ready!")

# ---------------- Chat ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask anything from your PDF...")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching..."):

            answer = ask(question)

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )