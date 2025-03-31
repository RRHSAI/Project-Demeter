import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import tempfile
import os

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
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
    
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
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

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to a temporary directory."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def load_pdf_documents(file_path):
    """Loads PDF documents from a file path."""
    try:
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

def chunk_documents(raw_documents):
    """Chunks the raw documents into smaller pieces."""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks, vector_store_path):
    """Indexes the document chunks into a vector store."""
    embeddings = OllamaEmbeddings(model=st.session_state.model)
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store_path

def find_related_documents(query, vector_store_path):
    """Finds related documents from the vector store."""
    embeddings = OllamaEmbeddings(model=st.session_state.model)
    vector_store = FAISS.load_local(vector_store_path, embeddings)
    return vector_store.similarity_search(query)

def generate_answer(user_query, context_documents):
    """Generates an answer to the user's query."""
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    language_model = OllamaLLM(model=st.session_state.model)
    response_chain = conversation_prompt | language_model
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

if 'vector_store_paths' not in st.session_state:
    st.session_state.vector_store_paths = {}

if 'model' not in st.session_state:
    st.session_state.model = "deepseek-r1:1.5b"

model_input = st.text_input("Ollama Model", value=st.session_state.model)
st.session_state.model = model_input

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    file_name = uploaded_pdf.name
    vector_store_path = f"vector_store_{file_name}"
    
    if file_name not in st.session_state.vector_store_paths:
        raw_docs = load_pdf_documents(saved_path)
        if raw_docs:
            processed_chunks = chunk_documents(raw_docs)
            st.session_state.vector_store_paths[file_name] = index_documents(processed_chunks, vector_store_path)
            st.success("✅ Document processed successfully! Ask your questions below.")

    user_input = st.chat_input("Enter your question about the document...")
    if user_input and file_name in st.session_state.vector_store_paths:
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input, st.session_state.vector_store_paths[file_name])
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)

    # Clean up temporary file
    if 'saved_path' in locals() and os.path.exists(saved_path):
        os.remove(saved_path)