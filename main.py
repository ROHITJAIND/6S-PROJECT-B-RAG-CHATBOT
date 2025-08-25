# main.py

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional

# Import your RAG functions
from rag_pipeline import process_pdf, get_answer_from_query

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered PDF Chatbot (RAG)",
    description="Upload a PDF and ask questions about its content.",
    version="1.0.0"
)

# --- Pydantic Models for Input/Output ---

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_context: str

class UploadResponse(BaseModel):
    message: str
    filename: str

# --- API Endpoints ---

@app.post("/upload-pdf/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file. The file is processed and indexed.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are accepted."
        )
        
    # Create a temporary directory to store the file
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF and create the vector store
        # In a real app, this should be an async background task
        process_pdf(file_path)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during file processing: {str(e)}"
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "message": "PDF processed and indexed successfully.",
        "filename": file.filename
    }


@app.post("/query/", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """
    Endpoint to ask a question about the uploaded PDF.
    """
    # For this simple example, we assume the index 'faiss_index' exists.
    # A more robust app would handle multiple documents and users.
    if not os.path.exists("faiss_index"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No PDF has been indexed yet. Please upload a PDF first."
        )

    try:
        result = get_answer_from_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        )