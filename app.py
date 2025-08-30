import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
import time
from dotenv import load_dotenv

import fitz  # PyMuPDF
import easyocr
from PIL import Image
import numpy as np
from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def process_pdf_and_create_vector_store(pdf_files, api_key):
    all_docs = []
    
    reader = easyocr.Reader(['en'])

    for pdf_file in pdf_files:
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            
            
            if len(text.strip()) < 100: 
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    ocr_results = reader.readtext(np.array(img))
                    
                    text = " ".join([result[1] for result in ocr_results])
                except Exception as e:
                    st.warning(f"Could not perform OCR on page {page_num + 1} of {pdf_file.name}. Skipping. Error: {e}")
                    text = "" 
            
            if text:
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_file.name, "page": page_num + 1}
                ))

    if not all_docs:
        st.error("Could not extract any text from the uploaded PDF(s). Please check the file(s).")
        return None

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = text_splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    vector_store = FAISS.from_documents(docs_chunks, embeddings)
    
    return vector_store

def create_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the provided context,
    just say, "The answer is not available in the context." Don't provide a wrong answer.
    If it ask to do any mathematical calculation like calculate the count the total subjects,
    or calculate the gpa please provide correct answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

st.set_page_config(page_title="Chat with your PDF 📄", page_icon="🤖")
st.title("🤖 Chat with your PDF 📄")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Upload your PDF")
    
    if not api_key:
        api_key = st.text_input("Please enter your Google API Key:", type="password")
        if api_key:
            st.success("API Key loaded!")
    
    uploaded_files = st.file_uploader(
        "Upload your PDF files and click 'Process'", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process"):
        if uploaded_files and api_key:
            with st.spinner("Processing your PDFs... this may take a moment."):
                try:
                    st.session_state.vector_store = process_pdf_and_create_vector_store(uploaded_files, api_key)
                    st.success("PDFs processed successfully!")
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
        elif not api_key:
            st.warning("Please enter your Google API Key first.")
        else:
            st.warning("Please upload at least one PDF file.")

st.markdown("### Chat History")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a PDF file before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                docs = st.session_state.vector_store.similarity_search(prompt)
                
                if not docs:
                    answer = "I'm sorry, I couldn't find any relevant information in the document for your question. Please try rephrasing it."
                    message_placeholder.markdown(answer)
                else:
                    chain = create_conversational_chain(api_key)
                    response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                    answer = response.get("output_text", "No answer found.")
                    
                    full_response = ""
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    with st.expander("View Source Context"):
                        source_context = "\n---\n".join([doc.page_content for doc in docs])
                        st.write(source_context)
                    
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"An error occurred: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})