# contains custom langchain tools for various tasks.
from src.memory.vectorizer import VectorRetriever

#Custom function to call when the LLM decides to apply for a vector search among the long term memory

def call_vector_search(query, vector_retriever: VectorRetriever):
    results = vector_retriever.similarity_search(query)
    #print("retrieved results: ", results)
    return [results[0].page_content for results in results] 


    