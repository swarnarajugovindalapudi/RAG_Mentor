from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_db(splits):
    """Converts text chunks into embeddings and stores them in ChromaDB."""
    # 1. Initialize open-source embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. Create and persist the vector database
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    return vectorstore