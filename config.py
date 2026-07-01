"""
config.py — single source of truth for all tunable settings.

Everything the rest of the app needs to know about *which* models to call,
*where* to store data, and *how* to chunk/retrieve lives here. Change a value
once and every module picks it up on the next run.

A few settings can be overridden by environment variables so the SAME code runs
unchanged locally and inside the Hugging Face Docker Space (which sets a
writable CHROMA_DIR). The defaults below are the local values, so if no env var
is set the behaviour is identical to before.
"""

import os

# --- Ollama / model settings ---------------------------------------------
# The LLM that powers the agent's reasoning AND the per-tool generation.
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
# The embedding model used to turn note chunks into vectors for Chroma.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
# Where the local Ollama server listens. Ollama's default is 11434. In the
# Docker Space, Ollama runs in the same container, so localhost still applies.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Low temperature -> more deterministic, factual answers (good for study notes).
TEMPERATURE = 0.2

# --- Vector store (Chroma) settings --------------------------------------
# Folder on disk where Chroma persists the collection between runs. On the
# Space this points at a writable path inside the container (storage is
# ephemeral there, which is fine for a demo).
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
# Name of the collection that holds all note chunks.
COLLECTION_NAME = "study_notes"

# --- Retrieval / chunking settings ---------------------------------------
# How many chunks to pull back from Chroma per query.
TOP_K = 4
# RecursiveCharacterTextSplitter settings. ~1000 chars per chunk with a 150
# char overlap keeps sentences/ideas from being cut awkwardly at boundaries.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
