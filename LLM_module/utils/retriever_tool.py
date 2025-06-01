from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_trends",
    "Search for the latest trends in fashion and hobbies and return relevant information.",
)

def retriever_node_function(state):
    """
    Retrieves documents based on the user query in the AgentState
    and updates the state with the retrieved context.
    """
    user_query = state.get("user_query")
    if not user_query:
        # Or handle this case as appropriate, maybe return an empty list of documents
        # or signal that no query was present.
        # For now, if there's no user_query in the state, we can't retrieve.
        print("No user query found in state for retrieval.")
        return {"context": []}

    # Assuming your retriever has an 'invoke' or similar method.
    # The exact method might depend on the retriever's implementation.
    # If 'retriever' is a BaseRetriever, it should have 'invoke' or 'get_relevant_documents'.
    try:
        documents = retriever.invoke(user_query)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        documents = []

    return {"context": documents}

# If you want to keep the original retriever_tool for other uses,
# you can leave it as is. Otherwise, you might not need it if
# the retriever_node_function is your primary way of using the retriever
# within the graph.