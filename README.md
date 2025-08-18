# 🤖 AI-Powered PDF Chatbot (RAG)

This project is a complete Retrieval-Augmented Generation (RAG) system that allows you to "chat" with your PDF documents. You can upload any PDF, and the application will index its content to answer your questions with high accuracy, providing context-aware responses and source references directly from the document.

The application is built with a **FastAPI** backend for the AI logic and a **Streamlit** frontend for a user-friendly chat interface.

## 🎯 Problem Statement

The objective is to build a backend that ingests any PDF, indexes it for semantic search, and answers queries via a RAG pipeline. This provides a powerful tool for quickly extracting information from dense documents without manual reading.

### Core Features

- **Data Processing & Retrieval**: Accept a PDF upload, extract its text, and split it into semantic chunks.
- **Vector Embeddings**: Generate embeddings for each text chunk and store them in a FAISS vector database for fast similarity search.
- **RAG Pipeline**: Implement a retrieval module to fetch the most relevant text chunks for a given query and use a Large Language Model (LLM) to generate a context-aware answer.
- **FastAPI Backend**: Provide robust REST API endpoints for document upload and querying, with input validation using Pydantic.

---

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI
- **Frontend Framework**: Streamlit
- **LLM Orchestration**: LangChain
- **LLM**: Google Gemini Pro
- **Embeddings Model**: `models/embedding-001` (Google Generative AI)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: PyPDF
- **API Interaction**: Requests, Uvicorn

---

## 📂 Project Structure

```
6S-PROJECT-B-RAG-CHATBOT/
├── main.py             # FastAPI backend server
├── rag_pipeline.py     # Core RAG logic (PDF processing, vector store, QA chain)
├── frontend.py         # Streamlit frontend application
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (for API keys)
└── README.md           # Project documentation (this file)
```

---

## deliverables

This repository contains all the necessary components to run the application locally:

1.  **FastAPI Backend Source (`main.py`, `rag_pipeline.py`)**:

    - Endpoint to upload and process a PDF (`/upload-pdf/`).
    - Endpoint to accept a user query and return a context-based answer (`/query/`).

2.  **Serialized Embedding Index (`faiss_index/`)**:

    - After a PDF is processed, a `faiss_index` directory is created locally. This contains the serialized vector store, allowing for persistent and fast retrieval without reprocessing the document.

3.  **README.md (`README.md`)**:
    - Detailed instructions for setup, local execution, API usage, and notes on the models and configuration used.

---

## 🚀 Setup and Local Run Instructions

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

- Python 3.9+
- A Google Gemini API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd rag-pdf-chatbot
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a file named `.env` in the root of the project directory and add your Google API key:

```
# .env file
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

### 6. Run the Application

You need to run the backend and frontend servers in two separate terminals.

**Terminal 1: Start the FastAPI Backend**

```bash
uvicorn main:app --reload
```

The backend will be available at `http://127.0.0.1:8000`.

**Terminal 2: Start the Streamlit Frontend**

```bash
streamlit run frontend.py
```

The frontend web interface will open automatically in your browser at `http://localhost:8501`.

---

## ⚙️ API Usage (via cURL)

You can also interact with the backend directly using any API client or `curl`.

### 1. Upload a PDF

Send a `POST` request with your PDF file to the `/upload-pdf/` endpoint.

```bash
curl -X POST -F "file=@/path/to/your/document.pdf" [http://127.0.0.1:8000/upload-pdf/](http://127.0.0.1:8000/upload-pdf/)
```

**Expected Response:**

```json
{
  "message": "PDF processed and indexed successfully.",
  "filename": "document.pdf"
}
```

### 2. Query the PDF

Send a `POST` request with your question in a JSON payload to the `/query/` endpoint.

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"question": "What are the main skills listed in this resume?"}' \
[http://127.0.0.1:8000/query/](http://127.0.0.1:8000/query/)
```

**Expected Response:**

```json
{
  "answer": "The main skills listed are Python, FastAPI, Machine Learning, and Natural Language Processing.",
  "source_context": "Relevant text chunks from the PDF that were used to generate the answer..."
}
```
