import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "./chroma_db"


def get_embedding_model(model_name=DEFAULT_EMBEDDING_MODEL):
    """Initialize HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vector_store(chunks, persist_directory=DEFAULT_PERSIST_DIR, collection_name="ragmentor_docs"):
    """Create a fresh Chroma vector store from ingested chunks with page_number metadata."""
    if not chunks:
        raise ValueError("No document chunks available to build vector store.")

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    documents = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        meta = dict(chunk.get("metadata") or {})
        page_num = meta.get("page_number", 1)
        meta["page_number"] = int(page_num) if str(page_num).isdigit() else 1
        documents.append(Document(page_content=text, metadata=meta))

    if not documents:
        raise ValueError("All chunks were empty after cleaning.")

    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectorstore


def load_vector_store(persist_directory=DEFAULT_PERSIST_DIR, collection_name="ragmentor_docs"):
    """Load an existing Chroma vector store if available."""
    if not os.path.exists(persist_directory):
        return None
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def retrieve_relevant_chunks(vectorstore, query, k=4):
    """Retrieve matching chunks along with sorted unique page numbers."""
    if vectorstore is None:
        return {
            "chunks": [],
            "page_numbers": [],
            "context": "",
        }

    query_text = (query or "").strip()
    if not query_text:
        return {
            "chunks": [],
            "page_numbers": [],
            "context": "",
        }

    docs = vectorstore.similarity_search(query_text, k=k)

    retrieved_chunks = []
    page_set = set()
    context_parts = []

    for doc in docs:
        page_num = doc.metadata.get("page_number", 1)
        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = 1

        page_set.add(page_num)
        retrieved_chunks.append(
            {
                "text": doc.page_content,
                "page_number": page_num,
                "metadata": doc.metadata,
            }
        )
        context_parts.append(f"[Page {page_num}] {doc.page_content}")

    page_numbers = sorted(page_set)
    context_text = "\n\n---\n\n".join(context_parts)

    return {
        "chunks": retrieved_chunks,
        "page_numbers": page_numbers,
        "context": context_text,
    }