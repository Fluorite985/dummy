# rag_system.py

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def load_model_and_db():
    """
    Loads the ChromaDB database and the Ollama LLM model.
    This function is designed to be cached by Streamlit's @st.cache_resource.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Ensure the model used here supports streaming. Ollama does.
    model = OllamaLLM(model="llama3.2") 
    return db, model

def query_rag_stream(query_text: str, db: Chroma, model: OllamaLLM):
    """
    Queries the RAG system and streams the response.

    This function is a generator that yields parts of the response as they
    are generated. It first yields the answer tokens and then yields the
    source documents.

    Yields:
        dict: A dictionary with either a 'token' key for the response text
              or a 'sources' key for the source documents.
    """
    # 1. Retrieve relevant documents from the database
    results = db.similarity_search_with_score(query_text, k=5)
    
    if not results:
        yield {"token": "I'm sorry, but I couldn't find any relevant information in the documents to answer your question."}
        yield {"sources": []}
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # 2. Format the prompt with the retrieved context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 3. Stream the response from the model
    # The model.stream() method returns an iterator of tokens
    for token in model.stream(prompt):
        yield {"token": token}

    # 4. After streaming the response, yield the sources
    sources = [doc.metadata for doc, _score in results]
    yield {"sources": sources}