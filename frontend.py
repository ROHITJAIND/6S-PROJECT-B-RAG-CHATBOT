import streamlit as st
import requests
import time

FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Chat with your PDF 📄",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🤖 Chat with your PDF 📄")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a PDF and I'll help you find answers within it."}]
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF file and click 'Process'", type=["pdf"]
    )

    if st.button("Process"):
        if uploaded_file is not None:
            with st.spinner("Processing your PDF... this may take a moment."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{FASTAPI_URL}/upload-pdf/", files=files, timeout=600)
                    if response.status_code == 201:
                        st.success("PDF processed successfully!")
                        st.session_state.file_uploaded = True
                        st.session_state.messages = [{"role": "assistant", "content": "PDF processed! You can now ask questions."}]
                    else:
                        st.error(f"Error processing PDF: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the backend: {e}")
        else:
            st.warning("Please upload a PDF file first.")

st.markdown("### Chat History")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.file_uploaded:
        st.warning("Please upload and process a PDF file before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    query_payload = {"question": prompt}
                    response = requests.post(f"{FASTAPI_URL}/query/", json=query_payload, timeout=600)
                    
                    if response.status_code == 200:
                        api_response = response.json()
                        answer = api_response.get("answer", "No answer found.")
                        
                        for chunk in answer.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)

                        with st.expander("View Source Context"):
                            st.write(api_response.get("source_context", "No source context available."))
                    else:
                        full_response = f"Error: {response.text}"
                        message_placeholder.markdown(full_response)

                except requests.exceptions.RequestException as e:
                    full_response = f"Could not connect to the backend: {e}"
                    message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

