"""
ingest.py — the RAG *storage* layer.

Responsibilities:
  1. Load PDF / .txt files into LangChain Documents.
  2. Split them into overlapping chunks.
  3. Embed the chunks with nomic-embed-text (via Ollama) and store them in a
     persistent Chroma collection.
  4. Provide retrieval + de-duplication + "clear all" helpers used by the agent
     tools and the UI.

The agent (agent.py) never touches Chroma directly — it goes through the
`retrieve()` helper here. That keeps the reasoning layer and the storage layer
cleanly separated.
"""

import os
from typing import List

import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


# --- Ollama helpers -------------------------------------------------------

def check_ollama() -> bool:
    """Quick health check: is the local Ollama server reachable?

    Used by the UI to show a friendly "start Ollama" message instead of an
    ugly stack trace when the server isn't running.
    """
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_embeddings() -> OllamaEmbeddings:
    """The embedding model wrapper. One place to construct it."""
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection.

    Chroma lazily creates the collection + on-disk folder if they don't exist,
    so calling this before any ingestion is safe — you just get an empty store.
    """
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


# --- Ingestion ------------------------------------------------------------

def _load_file(path: str) -> List[Document]:
    """Load a single file into Documents, choosing a loader by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext == ".txt":
        # utf-8 with autodetect fallback handles most study-note text files.
        return TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Upload .pdf or .txt.")


def ingest_files(paths: List[str]) -> str:
    """Load, split, embed, and store the given files. Returns a status summary.

    De-duplication: before adding a file's chunks we delete any existing chunks
    that share the same `source` filename. So re-uploading the same file
    *replaces* it rather than creating duplicates, while uploading a *new* file
    simply adds to the collection.
    """
    if not paths:
        return "No files selected. Please choose one or more .pdf / .txt files."

    if not check_ollama():
        return ("⚠️ Ollama isn't reachable, so embeddings can't be created.\n"
                "Start it with `ollama serve` and make sure the models are pulled.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    vectorstore = get_vectorstore()

    summary_lines = []
    for path in paths:
        filename = os.path.basename(path)
        try:
            docs = _load_file(path)
        except Exception as exc:  # bad/corrupt file, unsupported type, etc.
            summary_lines.append(f"❌ {filename}: {exc}")
            continue

        chunks = splitter.split_documents(docs)
        if not chunks:
            summary_lines.append(f"⚠️ {filename}: no text extracted (empty or scanned image?).")
            continue

        # Tag every chunk with its source so we can de-dup and cite it later.
        for chunk in chunks:
            chunk.metadata["source"] = filename

        # De-dup: remove prior chunks from this same file, then add fresh ones.
        try:
            vectorstore.delete(where={"source": filename})
        except Exception:
            # Deleting from an empty collection can raise on some Chroma
            # versions; it's harmless, so we ignore it.
            pass

        vectorstore.add_documents(chunks)
        summary_lines.append(f"✅ {filename}: {len(chunks)} chunks embedded.")

    total = collection_count()
    summary_lines.append(f"\n📚 Collection now holds {total} chunks total.")
    return "\n".join(summary_lines)


# --- Retrieval & maintenance ---------------------------------------------

def retrieve(query: str, k: int = config.TOP_K) -> List[Document]:
    """Return the top-k most relevant chunks for a query.

    Returns an empty list if the collection is empty or nothing matches, so
    callers can honestly say "I don't have notes on that."
    """
    vectorstore = get_vectorstore()
    if collection_count() == 0:
        return []
    return vectorstore.similarity_search(query, k=k)


def collection_count() -> int:
    """How many chunks are currently stored. Used for empty-state checks."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def clear_all() -> str:
    """Wipe every chunk from the collection (drives the 'Clear all notes' button)."""
    vectorstore = get_vectorstore()
    try:
        # Fetch all stored ids, then delete them. Simpler and more portable
        # across Chroma versions than dropping the whole collection object.
        existing = vectorstore._collection.get()
        ids = existing.get("ids", [])
        if ids:
            vectorstore._collection.delete(ids=ids)
        return "🧹 Cleared all notes from the collection."
    except Exception as exc:
        return f"Could not clear collection: {exc}"
