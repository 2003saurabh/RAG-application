import streamlit as st
import time
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM 
import plotly.express as px
import pandas as pd

# Custom CSS with both dark and light theme support
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = 0

PROMPT_TEMPLATE = """
You are an expert research assistant. Provide concise, factual answers based on the provided context.
If unsure, state that you don't know.

Query: {user_query}
Context: {document_context}
Answer:
"""


PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Utility Functions
def check_rate_limit():
    """Implement basic rate limiting"""
    current_time = time.time()
    if current_time - st.session_state.last_query_time < 1:
        st.warning("Please wait before making another query")
        return False
    st.session_state.last_query_time = current_time
    return True

def sanitize_input(user_input):
    """Basic input sanitization"""
    return re.sub(r'[<>{}]', '', user_input)

@st.cache_data
def save_uploaded_file(uploaded_file):
    """Cache and save uploaded file"""
    try:
        file_path = PDF_STORAGE_PATH + uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

@st.cache_data
def load_pdf_documents(file_path):
    """Cache and load PDF documents"""
    try:
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

def chunk_documents(raw_documents, chunk_size=1000, chunk_overlap=200):
    """Process documents into chunks"""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def process_in_batches(documents, batch_size=5):
    """Process documents in batches"""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        DOCUMENT_VECTOR_DB.add_documents(batch)

def show_document_stats(docs, chunks):
    """Display document statistics"""
    st.sidebar.markdown("### Document Statistics")
    st.sidebar.metric("Total Pages", len(docs))
    st.sidebar.metric("Total Chunks", len(chunks))
    st.sidebar.metric("Queries Made", st.session_state.query_count)

def generate_answer(user_query, context_documents):
    """Generate AI response"""
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    
    # Remove think tags and clean the response
    cleaned_response = response.replace('<think>', '').replace('</think>', '')
    return cleaned_response.strip()


def validate_pdf(uploaded_file):
    """Validate uploaded PDF file"""
    if uploaded_file is None:
        return False
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
        st.error("File too large. Please upload a smaller file (max 10MB).")
        return False
    if not uploaded_file.name.lower().endswith('.pdf'):
        st.error("Please upload a PDF file.")
        return False
    return True

# Main UI
st.title("📘 ClariDoc AI")
st.markdown("### Turning Documents into Insights.")

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    st.slider("AI Temperature", 0.0, 1.0, 0.7)
    if st.button("Clear Cache"):
        st.cache_data.clear()
        DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
        st.success("Cache cleared!")

# Main Interface
tab1, tab2 = st.tabs(["Chat Interface", "Document Analysis"])

with tab1:
    uploaded_pdf = st.file_uploader(
        "Upload Research Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )

    if uploaded_pdf and validate_pdf(uploaded_pdf):
        saved_path = save_uploaded_file(uploaded_pdf)
        if saved_path:
            raw_docs = load_pdf_documents(saved_path)
            if raw_docs:
                processed_chunks = chunk_documents(raw_docs, chunk_size=chunk_size)
                process_in_batches(processed_chunks)
                show_document_stats(raw_docs, processed_chunks)
                
                st.success("✅ Document processed successfully! Ask your questions below.")
                
                user_input = st.chat_input("Enter your question about the document...")
                
                if user_input and check_rate_limit():
                    user_input = sanitize_input(user_input)
                    st.session_state.query_count += 1
                    
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    with st.spinner("Analyzing document..."):
                        relevant_docs = DOCUMENT_VECTOR_DB.similarity_search(user_input)
                        ai_response = generate_answer(user_input, relevant_docs)
                        
                        # Store in chat history
                        st.session_state.chat_history.append({"user": user_input, "assistant": ai_response})
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(ai_response)
                        
                    

with tab2:
    if len(st.session_state.chat_history) > 0:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.text("User: " + chat["user"])
            st.text("Assistant: " + chat["assistant"])
            st.markdown("---")
        
        if st.button("Export Chat History"):
            chat_text = "\n\n".join([f"User: {chat['user']}\nAssistant: {chat['assistant']}" 
                                   for chat in st.session_state.chat_history])
            st.download_button(
                "Download Chat History",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )
    else:
        st.info("No chat history available yet. Start a conversation in the Chat Interface!")
