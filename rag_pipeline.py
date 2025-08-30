import os
from dotenv import load_dotenv
import fitz
import easyocr
from PIL import Image
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

try:
    ocr_reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    ocr_reader = None

def process_pdf(pdf_path: str):
    all_docs = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            if len(text.strip()) < 100 and ocr_reader:
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_results = ocr_reader.readtext(np.array(img))
                    text = " ".join([result[1] for result in ocr_results])
                except Exception as e:
                    print(f"OCR failed on page {page_num + 1}: {e}")
            if text:
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
                ))
        pdf_document.close()
        if not all_docs:
            raise ValueError("Could not extract any text from the PDF.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_chunks = text_splitter.split_documents(all_docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs_chunks, embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        raise e

def create_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the provided context,
    just say, "The answer is not available in the context." Don't provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_answer_from_query(query: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query)
    if not docs:
        return {"answer": "I'm sorry, I couldn't find any relevant information in the document for your question. Please try rephrasing it.", "source_context": ""}
    chain = create_conversational_chain()
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    source_context = "\n---\n".join([f"Page {doc.metadata.get('page', 'N/A')}:\n" + doc.page_content for doc in docs])
    return {"answer": response["output_text"], "source_context": source_context}
