"""
config.py — single source of truth for all tunable settings.

Everything the rest of the app needs to know about *which* models to call,
*where* to store data, and *how* to chunk/retrieve lives here. Change a value
once and every module picks it up on the next run.

Model backends:
  - The LLM (agent routing + per-tool generation) runs on **OpenRouter**, using
    its OpenAI-compatible API. Pick any tool-calling-capable free-tier model.
  - Embeddings run **locally** with a small sentence-transformers model — it
    downloads once, needs no server and no API key, and runs fine on CPU.
    (OpenRouter does not offer an embeddings endpoint, so embeddings can't come
    from there.)

A few settings can be overridden by environment variables so the SAME code runs
unchanged locally and inside a Hugging Face Docker Space. The one secret you
MUST provide is OPENROUTER_API_KEY.
"""

import os

# Optional: load a local .env file if python-dotenv is installed. This lets you
# keep OPENROUTER_API_KEY in a .env file for local dev. In production (e.g. a HF
# Space) the key is provided as a real environment variable / secret instead.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- LLM settings (OpenRouter) -------------------------------------------
# The LLM that powers the agent's reasoning AND the per-tool generation.
# NOTE: this is a *tool-calling* agent, so the model MUST support function/tool
# calling. Good free-tier options on OpenRouter (all support tools):
#   meta-llama/llama-3.3-70b-instruct:free
#   qwen/qwen-2.5-72b-instruct:free
#   mistralai/mistral-small-3.1-24b-instruct:free
# Free models can be rate-limited or rotate over time — swap via LLM_MODEL.
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen-2.5-72b-instruct:free")
# OpenRouter's OpenAI-compatible endpoint.
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
# Your OpenRouter API key. REQUIRED. Get one free at https://openrouter.ai/keys
# Set it as an env var locally (or in a .env file) and as a Space secret on HF.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Optional attribution headers OpenRouter uses for its rankings (harmless if unset).
OPENROUTER_APP_URL = os.environ.get("OPENROUTER_APP_URL", "")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "Study Companion")
# Low temperature -> more deterministic, factual answers (good for study notes).
TEMPERATURE = 0.2

# --- Embedding settings (local sentence-transformers) --------------------
# Runs in-process via HuggingFace sentence-transformers. all-MiniLM-L6-v2 is
# small (~90MB), fast on CPU, and produces 384-dim vectors. Downloaded once from
# the HF hub, then cached. No server, no API key.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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


def has_llm_key() -> bool:
    """True if an OpenRouter API key is configured. Used for friendly guards."""
    return bool(OPENROUTER_API_KEY.strip())
