from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv
import os
import shutil
import stat
import glob
from typing import List, Optional
import pdfParser  # Import our custom PDF parser

load_dotenv()

API_KEY = os.getenv("OPENAI_KEY")
CHROMADATAPATH = 'chromaDb'

def load_documents(DATA_PATH: str, UseAiLoader: bool = True) -> List[Document]:
    """
    Load documents from a directory with improved PDF parsing
    
    Args:
        DATA_PATH: Path to directory containing PDF files
        UseAiLoader: Whether to use the advanced AI-based loader
        
    Returns:
        List of Document objects
    """
    documents = []
    
    # Find all PDF files in the directory
    pdf_files = glob.glob(os.path.join(DATA_PATH, "**/*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_PATH}")
        return documents
    
    print(f"Found {len(pdf_files)} PDF files")
    
    if UseAiLoader:
        # Use our custom advanced PDF parser
        for pdf_path in pdf_files:
            print(f"Processing {pdf_path}...")
            
            # Create a temporary output file for the cleaned text
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            temp_output = f"temp_{base_name}_cleaned.txt"
            
            # Process the PDF and get cleaned text
            cleaned_text = pdfParser.process_pdf(pdf_path, temp_output)
            
            # Read the cleaned text file
            with open(temp_output, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create a Document object with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": 0,  # Since we've merged pages in our advanced parser
                    "filename": os.path.basename(pdf_path)
                }
            )
            documents.append(doc)
            
            # Clean up temporary file
            os.remove(temp_output)
    else:
        # Use the original PyPDFDirectoryLoader
        document_load = PyPDFDirectoryLoader(DATA_PATH)
        documents = document_load.load()
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding"""
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", " "]
    )
    return txt_splitter.split_documents(documents)

def get_embed_function():
    """Get the embedding function"""
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=API_KEY)

def add_to_chroma(chunks: List[Document]):
    """Add document chunks to Chroma vector store"""
    db = Chroma(
        persist_directory=CHROMADATAPATH,
        embedding_function=get_embed_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        if new_chunks:
            print("🧠 Sample stored chunk:", new_chunks[0].page_content[:200], "...")
    else:
        print("✅ No new documents to add")
    
    return db

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Calculate unique IDs for each chunk"""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def on_rm_error(func, path, exc_info):
    """Handle permission errors when removing files"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_database():
    """Clear the existing vector database"""
    if os.path.exists(CHROMADATAPATH):
        shutil.rmtree(CHROMADATAPATH, onerror=on_rm_error)
        print("✅ Chroma DB cleared.")
    else:
        print("ℹ️ Chroma DB not found.")

if __name__ == '__main__':
    # clear_database()
    
    # Load documents with the advanced parser
    allTextbooks = load_documents('data', UseAiLoader=True)
    
    # Split into chunks
    chunks = split_documents(allTextbooks)
    
    # Print a sample chunk
    if chunks:
        print("\nSample chunk:")
        print(f"Source: {chunks[0].metadata.get('source')}")
        print(f"Content preview: {chunks[0].page_content[:300]}...\n")
    
    # Add to vector database
    add_to_chroma(chunks)
    print("✅ Processing complete")