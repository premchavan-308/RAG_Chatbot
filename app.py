import streamlit as st
import tempfile

from ingest import create_vector_db
from rag import ask


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Chatbot")

st.write("Upload any PDF and ask questions from it.")

# ---------------- Session State ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# ---------------- Upload PDF ---------------- #

uploaded_file = st.file_uploader(
    "📄 Drag and Drop your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # If a new PDF is uploaded
    if st.session_state.current_pdf != uploaded_file.name:

        st.session_state.current_pdf = uploaded_file.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Creating Vector Database..."):

            st.session_state.db = create_vector_db(pdf_path)

        st.session_state.db_ready = True

        st.session_state.messages = []

        st.success("✅ Vector Database Ready!")

# ---------------- Display Chat ---------------- #

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- Chat ---------------- #

question = st.chat_input("Ask anything from your PDF...")

if question:

    if not st.session_state.db_ready:

        st.warning("Please upload a PDF first.")

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching document..."):

                answer = ask(question)

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("About")

    st.write("✅ Upload any PDF")
    st.write("✅ Automatic Vector Database")
    st.write("✅ Mistral Embeddings")
    st.write("✅ MMR Retrieval")
    st.write("✅ Mistral LLM")
