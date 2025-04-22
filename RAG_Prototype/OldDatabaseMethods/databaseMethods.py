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

load_dotenv()

API_KEY = os.getenv("OPENAI_KEY")
CHROMADATAPATH = 'chromaDb'

def load_documents(DATA_PATH: str):
    document_load = PyPDFDirectoryLoader(DATA_PATH)
    return document_load.load()

def split_documents(documents: list[Document]):
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", " "]
    )
    return txt_splitter.split_documents(documents)

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
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
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
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_database():
    if os.path.exists(CHROMADATAPATH):
        shutil.rmtree(CHROMADATAPATH, onerror=on_rm_error)
        print("✅ Chroma DB cleared.")
    else:
        print("Chroma DB not found.")

# if __name__ == '__main__':
#     clear_database()
#     allTextbooks = load_documents('data')
#     chunks = split_documents(allTextbooks)
#     print(chunks[0])
#     add_to_chroma(chunks)

# Testing chunks
if __name__ == '__main__':
    # clear_database()

    allTextbooks = load_documents('data')
    chunks = split_documents(allTextbooks)

    print(f"\n🔍 Total Chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[20:]):  # limit to first 10 for readability
        print(f"--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Page: {chunk.metadata.get('page')}")
        print(f"Content:\n{chunk.page_content}\n")
        print("-" * 40)