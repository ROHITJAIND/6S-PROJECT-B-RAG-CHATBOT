import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from rag_pipeline import process_pdf, get_answer_from_query

app = FastAPI(
    title="AI-Powered PDF Chatbot (RAG)",
    description="Upload a PDF and ask questions about its content.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_context: str

class UploadResponse(BaseModel):
    message: str
    filename: str

@app.post("/upload-pdf/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are accepted."
        )
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        process_pdf(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during file processing: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    return {
        "message": "PDF processed and indexed successfully.",
        "filename": file.filename
    }

@app.post("/query/", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
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
