from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, field_validator
from langchain_ollama import OllamaLLM
from pdfPlumberDatabase import get_embed_function
from langchain_chroma import Chroma

model = OllamaLLM(model="phi4-mini", temperature=0.0)
CHROMADATAPATH = "chromaDb"

class Queries(BaseModel):
    subQueries: list

    @field_validator("subQueries")
    def ensure5orLess(cls, value):
        if (len(value) > 5):
            raise ValueError(f"More Than 5 Sub Queries Given: {value}\nLength of given value: {len(value)}")
        for val in value:
            if not isinstance(val, str):
                raise ValueError(f"Not all values are strings. Messed up value: {val}\nEntire list: {value}")
        return value

parser = PydanticOutputParser(pydantic_object=Queries)

prompt = PromptTemplate(
    template="""Seperate the query into a list of up to 5 STRING DATATYPE questions that would provide information solely related to the query. This is to be used in a RAG model to retrieve all related topics through the process of multi-querying.
    For example, you could seperate the input query: \"Compare and contrast the differences between water and hydrochloric acid\" into the following output [\"What are the attributes of water?\", \"What are the attributes of hydrochloric acid\"] 
    Only return a JSON object that matches the following format:\n{format_instructions}\n{query}\n""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def get_sub_queries_and_chunks(userQuery: str, k: int = 3):
    # Generating Sub Queries
    prompt_and_model = prompt | model
    queryData = {"query": userQuery}
    output = prompt_and_model.invoke(queryData)
    response = parser.invoke(output)
    response.subQueries.append(userQuery)  # including the original query

    print("Generated sub-queries:")
    print(response.subQueries)

    # Step 2: Setup vector DB
    embedding_function = get_embed_function()
    vectordb = Chroma(persist_directory=CHROMADATAPATH, embedding_function=embedding_function)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # Step 3: Retrieve top chunks for each sub-query
    sub_query_chunks = {}
    for sub_query in response.subQueries:
        docs = retriever.get_relevant_documents(sub_query)
        sub_query_chunks[sub_query] = docs

    return sub_query_chunks


if (__name__=="__main__"):
    chunks = get_sub_queries_and_chunks("Which process removes a hydroxyl group (-OH) from one monomer and a hydrogen atom (-H) from the other, forming a covalent bond between the two monomers and a molecule of water?")
    allChunksCombined = []
    for subQuery in chunks:
        for chunk in chunks[subQuery]:
            allChunksCombined.append(chunk)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in allChunksCombined])
    sources = [doc.metadata.get("id", None) for doc in allChunksCombined]
    pageNum = [doc.metadata.get("page", None) for doc in allChunksCombined]

    print(context_text)
    print(sources)
    print(pageNum)    # testing = Queries(subQueries=["1", "2", "3", "4", "5", "6"])
    # print(testing)