# 📚 RAG Chatbot using Mistral AI

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDF documents and ask natural language questions. The application retrieves the most relevant document chunks using **ChromaDB** and **Mistral Embeddings**, then generates context-aware answers using **Mistral AI**.

---

## 🚀 Features

* 📄 Drag & Drop PDF Upload
* 🤖 AI-powered Question Answering
* 🔍 Semantic Search with ChromaDB
* 🧠 Mistral AI for Response Generation
* 📚 Mistral Embeddings for Vector Search
* ✂️ Automatic Document Chunking
* 💬 Interactive Chat Interface using Streamlit
* ⚡ Fast and Lightweight RAG Pipeline

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Mistral AI**
* **Mistral Embeddings**
* **ChromaDB**
* **PyPDF**
* **Python Dotenv**

---

## 📂 Project Structure

```
RAG-Project/
│
├── app.py              # Streamlit application
├── ingest.py           # PDF ingestion & vector database creation
├── rag.py              # Retrieval and response generation
├── requirements.txt
├── README.md
├── .gitignore

```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/RAG-Project.git
cd RAG-Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create a `.env` file

```env
MISTRAL_API_KEY=your_api_key_here
```

Get your API key from the Mistral AI platform.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser and navigate to:

```
http://localhost:8501
```

---

## 📖 How It Works

1. Upload a PDF document.
2. The PDF is loaded using **PyPDFLoader**.
3. The document is split into smaller chunks.
4. Mistral Embeddings convert each chunk into vector representations.
5. ChromaDB stores the vectors.
6. User questions are matched with the most relevant document chunks.
7. Mistral AI generates an answer based only on the retrieved context.

---

## 🔮 Future Enhancements

* Support multiple PDF documents
* Chat history with memory
* Display source page numbers
* Citation-based responses
* Streaming AI responses
* Conversation export
* Dark mode UI

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository, create a new branch, and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Prem Ramdas Chavan**

GitHub: https://github.com/premchavan-308
