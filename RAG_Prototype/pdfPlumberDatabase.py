from langchain_community.document_loaders import PyPDFDirectoryLoader # Faster but also keeps random slop such as headers, footers, trademarks, etc.
from langchain_community.document_loaders import UnstructuredPDFLoader # Slower but higher quality + ai tools
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import shutil
import stat
import pdfplumber
import re

load_dotenv()

API_KEY = os.getenv("OPENAI_KEY")
CHROMADATAPATH = 'chromaDb'

def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Cleaning up text by removing instances of certain things
        if not line:
            continue
        if re.search(r'(copyright|all rights reserved|trademark)', line, re.I):
            continue
        if re.search(r'https?://', line):
            continue
        if re.match(r'^Page\s+\d+', line, re.I):
            continue
        if len(line) < 5:  # Skipping super short lines (junk lines)
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def load_clean_pdfs_no_tables(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if not filename.endswith('.pdf'):
            continue

        path = os.path.join(directory, filename)
        print(f"PROCESSING FILE: {filename}")

        with pdfplumber.open(path) as pdf:
            # Create one document for the entire file
            full_text = []
            page_ranges = []  # To track which ranges belong to which pages
            current_char_count = 0
            
            for page_number, page in enumerate(pdf.pages, 1):
                # Removing detected tables
                if page.extract_tables():
                    print(f"SKIPPING TABLE ON PAGE: {page_number + 1}")
                    continue # Skips the table on the page

                # Getting plain text (images are ignored by default)
                text = page.extract_text()
                if text:
                    cleaned = clean_text(text)
                    start_pos = current_char_count
                    current_char_count += len(cleaned)
                    
                    # Storing character range for this page
                    page_ranges.append({
                        "page": page_number,
                        "start": start_pos,
                        "end": current_char_count
                    })
                    
                    full_text.append(cleaned)
            
            if full_text:
                combined_text = "\n\n".join(full_text)
                doc = Document(
                    page_content=combined_text,
                    metadata={
                        "source": path,
                        "filename": filename,
                        "page_ranges": page_ranges
                    }
                )
                documents.append(doc)

    return documents


# def load_documents(DATA_PATH: str):
#     document_load = PyPDFDirectoryLoader(DATA_PATH)
#     return document_load.load()

def load_documents(DATA_PATH: str):
    return load_clean_pdfs_no_tables(DATA_PATH)

def split_documents(documents: list[Document]):
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "."]
    )
    
    all_chunks = []
    
    for doc in documents:
        # Split the document into chunks
        chunks = txt_splitter.create_documents(
            [doc.page_content], 
            [{"source": doc.metadata.get("source"), "filename": doc.metadata.get("filename")}]
        )
        
        # For each chunk, determine which page it belongs to
        page_ranges = doc.metadata.get("page_ranges", [])
        
        for chunk in chunks:
            # Find the first character position of this chunk in the original document
            chunk_text = chunk.page_content
            chunk_start = doc.page_content.find(chunk_text)
            
            if chunk_start == -1:  # If exact match not found due to overlap adjustments
                # Approximate by using the first 50 characters
                chunk_prefix = chunk_text[:50]
                chunk_start = doc.page_content.find(chunk_prefix)
                if chunk_start == -1:
                    print("CHUNK NOT FOUND, DEFAULTING TO BEGINNING")
                    chunk_start = 0  # Default to beginning if still not found (shouldn't happen)
            
            # Find which page contains this chunk's start position
            assigned_page = 1  # Default to page 1
            for page_info in page_ranges:
                if chunk_start >= page_info["start"] and chunk_start < page_info["end"]:
                    assigned_page = page_info["page"]
                    break
            
            # Update the chunk's metadata with the page number
            chunk.metadata["page"] = assigned_page
            all_chunks.append(chunk)
    
    return all_chunks


def get_embed_function():
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=API_KEY)

def add_to_chroma(chunks: list[Document]):
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
    else:
        print("✅ No new documents to add")
    print("Storing chunk:", chunk.page_content[:2000], "...")
    return db

def calculate_chunk_ids(chunks):
    # Grouping chunks by source
    source_chunks = {}
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source not in source_chunks:
            source_chunks[source] = []
        source_chunks[source].append(chunk)
    
    # For each source, assign sequential IDs with page numbers included
    for source, source_chunks_list in source_chunks.items():
        # Sort chunks by page number first
        source_chunks_list.sort(key=lambda x: x.metadata.get("page", 0))
        
        last_page = None
        chunk_index = 0
        
        for chunk in source_chunks_list:
            page = chunk.metadata.get("page", 0)
            
            # Reset chunk index when page changes
            if page != last_page:
                chunk_index = 0
                last_page = page
            
            # Creating unique id for each chunk based on the source text, the page number, and the chunk index
            chunk_id = f"{source}:{page}:{chunk_index}"
            chunk.metadata["id"] = chunk_id
            
            chunk_index += 1
    
    return chunks

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_database():
    if os.path.exists(CHROMADATAPATH):
        shutil.rmtree(CHROMADATAPATH, onerror=on_rm_error)
        print("✅ Chroma DB cleared.")
    else:
        print("Chroma DB not found.")

if __name__ == '__main__':
    # clear_database()
    allTextbooks = load_documents('data')
    chunks = split_documents(allTextbooks)
    print(chunks[0])
    add_to_chroma(chunks)

# Testing chunks
# if __name__ == '__main__':
#     # clear_database()

#     allTextbooks = load_documents('data')
#     chunks = split_documents(allTextbooks)

#     print(f"\n🔍 Total Chunks: {len(chunks)}\n")

#     for i, chunk in enumerate(chunks[20:]):
#         print(f"--- Chunk {i + 1} ---")
#         print(f"Source: {chunk.metadata.get('source')}")
#         print(f"Page: {chunk.metadata.get('page')}")
#         print(f"Content:\n{chunk.page_content}\n")
#         print("-" * 40)