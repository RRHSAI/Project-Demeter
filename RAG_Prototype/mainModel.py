#TODO Add feature where page number can be given to the user (maybe only give steps to solution, rather than act solution)
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pdfPlumberDatabase import get_embed_function
from langchain.retrievers.multi_query import MultiQueryRetriever
from customPromptMultiQuery import get_sub_queries_and_chunks
import logging
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

CHROMADATAPATH = 'chromaDb'
PROMPT = """
You are an AI tutor, answer the question a student just asked you BASED ONLY ON THE FOLLOWING CONTEXT:

{context}

---

Answer the question based on the above context: {question}
"""

MODEL = OllamaLLM(model="phi4-mini")

def single_query(query_text: str):
    # Preparing the database
    embedding_function = get_embed_function()
    db = Chroma(persist_directory=CHROMADATAPATH, embedding_function=embedding_function)

    # Searching the data
    results = db.similarity_search_with_score(query_text, k=5) # Gets the top 5 most relevant pieces of data

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    response_text = MODEL.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    pageNum = [newDoc.metadata.get("page", None) for newDoc, _newScore in results]
    
    formatted_response = f"Response: {response_text}\nSources: {sources}\nPage Numbers Found: {pageNum}"
    print(formatted_response)
    return response_text

def multi_query_default(original_query: str):
    embedding_function = get_embed_function()

    vectordb = Chroma(persist_directory=CHROMADATAPATH, embedding_function=embedding_function)
    retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}), llm=MODEL # setting retriever to get top 3 chunks (default is 5)
    )

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    unique_docs = retriever_from_llm.invoke(original_query)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])
    # print(context_text)
    # print(len(unique_docs))

    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context_text, question=original_query)
    print(prompt)

    response_text = MODEL.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc in unique_docs]
    pageNum = [doc.metadata.get("page", None) for doc in unique_docs]
    
    formatted_response = f"Response: {response_text}\nSources: {sources}\nPage Numbers Found: {pageNum}"
    print(formatted_response)
    return response_text, {'Sources': sources, 'Page Nums': pageNum}

def multi_query_custom(user_query: str):

    chunks = get_sub_queries_and_chunks(user_query)
    allChunksCombined = []
    for subQuery in chunks:
        for chunk in chunks[subQuery]:
            allChunksCombined.append(chunk)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in allChunksCombined])
    sources = [doc.metadata.get("id", None) for doc in allChunksCombined]
    pageNum = [doc.metadata.get("page", None) for doc in allChunksCombined]

    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context_text, question=user_query)
    print(prompt)

    response_text = MODEL.invoke(prompt)
    
    formatted_response = f"Response: {response_text}\nSources: {sources}\nPage Numbers Found: {pageNum}"
    print(formatted_response)
    return response_text

def main():
    # The argparse stuff is just so you can run it in console with a string parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    # single_query(query_text)
    multi_query_custom(query_text)

if __name__ == '__main__':
    main()