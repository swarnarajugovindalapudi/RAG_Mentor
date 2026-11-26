import os
import tempfile
import uuid
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function(model_name=DEFAULT_EMBEDDING_MODEL):
    """Return HuggingFace sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vector_store(
    chunks,
    persist_directory=DEFAULT_PERSIST_DIR,
    collection_name=None,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
):
    """Build a Chroma vector store safely without Windows file-lock conflicts."""
    if not chunks:
        raise ValueError("No text chunks available to index.")

    embeddings = get_embedding_function(embedding_model)
    safe_collection = collection_name or f"ragmentor_{uuid.uuid4().hex[:10]}"
    target_dir = os.path.join(persist_directory, safe_collection)
    os.makedirs(target_dir, exist_ok=True)

    try:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=target_dir,
            collection_name=safe_collection,
        )
    except PermissionError:
        fallback_dir = tempfile.mkdtemp(prefix="ragmentor_chroma_")
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=fallback_dir,
            collection_name=safe_collection,
        )


def retrieve_relevant_chunks(vectorstore, query, k=4):
    """Retrieve top-k chunks and return combined context plus sorted unique page numbers."""
    if vectorstore is None or not (query or "").strip():
        return {"context": "", "page_numbers": [], "docs": []}

    docs = vectorstore.similarity_search(query.strip(), k=k)
    page_numbers = sorted(
        {
            doc.metadata.get("page")
            for doc in docs
            if doc.metadata and doc.metadata.get("page") is not None
        }
    )

    context_parts = []
    for doc in docs:
        page_num = doc.metadata.get("page") if doc.metadata else None
        prefix = f"[Page {page_num}] " if page_num is not None else ""
        context_parts.append(f"{prefix}{doc.page_content.strip()}")

    return {
        "context": "\n\n".join(context_parts),
        "page_numbers": page_numbers,
        "docs": docs,
    }