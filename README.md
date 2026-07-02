---
title: Study Companion
emoji: 📖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 📖 Study Companion

A **LangChain agent** that answers questions, generates quizzes, and explains
concepts — grounded in *your own* study notes. The LLM runs on
[OpenRouter](https://openrouter.ai)'s free tier; embeddings run **locally** on
your machine (no embedding API needed).

The key idea: this is an **agent, not a fixed RAG chain**. The LLM decides
*which tool* to call for each message, and the UI shows you that decision live.

---

## Architecture

```
you ──▶ Gradio UI (app.py)
             │  your message
             ▼
     Tool-calling agent (agent.py)  ◀── OpenRouter LLM routes to ONE tool
             │
   ┌─────────┼──────────────┐
   ▼         ▼              ▼
answer_    make_quiz    explain_
from_notes             concept
   │         │              │
   └────── retrieve() ──────┘        (each tool does its own RAG)
             │
             ▼
     Chroma vector store (ingest.py)  ◀── local sentence-transformers embeddings
             │
             ▼
     your uploaded .pdf / .txt notes
```

**Why two different backends?** OpenRouter is a chat-completions service — it has
no embeddings endpoint. So the LLM comes from OpenRouter, while embeddings run
in-process with a small `sentence-transformers` model (free, no server, no key).

### The three tools
| Tool | When the agent uses it | What it returns |
|------|------------------------|-----------------|
| `answer_from_notes` | factual questions ("What is X?") | grounded answer + source snippet |
| `make_quiz` | "quiz me on…", "test me" | N mixed MCQ + short-answer questions **with an answer key** |
| `explain_concept` | "explain… simply", confusion | plain-language explanation + one analogy |

Each tool retrieves the top-k relevant chunks from Chroma and generates its
answer **only** from those chunks, so responses stay grounded in your notes.

### Files
| File | Role |
|------|------|
| `config.py` | all settings: model names, OpenRouter key/URL, paths, `k`, chunk size |
| `ingest.py` | load → split → embed (local) → persist Chroma; retrieval & de-dup |
| `agent.py` | the three tools + the tool-calling agent loop (OpenRouter LLM) |
| `app.py` | Gradio UI (upload, chat, live agent-step display) |

---

## Setup

### 1. Install Python dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Get an OpenRouter API key
1. Sign up at <https://openrouter.ai> and create a key at
   <https://openrouter.ai/keys> (the free tier is enough).
2. Make it available to the app. Either:
   - copy `.env.example` to `.env` and paste your key in, **or**
   - set it in your shell:
     ```bash
     # macOS/Linux
     export OPENROUTER_API_KEY=sk-or-...
     # Windows PowerShell
     $env:OPENROUTER_API_KEY="sk-or-..."
     ```

The **first run** downloads the local embedding model (~90 MB) once; after that
it's cached.

### 3. Run the app
```bash
python app.py
```
Open the local URL it prints (usually `http://127.0.0.1:7860`).

---

## How to use
1. **Upload** one or more `.pdf` / `.txt` notes on the left and click
   **📥 Ingest notes**. The status box shows how many chunks were embedded.
   *(A `sample_notes.txt` on photosynthesis is included so you can try it
   immediately.)*
2. **Ask** on the right. Try:
   - *"What is mitosis?"* → routes to `answer_from_notes`
   - *"Quiz me on the water cycle"* → routes to `make_quiz`
   - *"Explain entropy simply"* → routes to `explain_concept`
3. Watch the **🛠 Agent steps & sources** panel to see which tool the agent
   chose and the source snippet it grounded on.

Re-uploading the same file **replaces** its chunks (no duplicates); uploading a
new file **adds** to the collection. Use **🧹 Clear all notes** to start fresh.

---

## Choosing a model
Set `LLM_MODEL` (env var or `.env`) to any **tool-calling-capable** OpenRouter
model. This is a tool-calling agent, so the model *must* support function calling.
Good free options:
- `meta-llama/llama-3.3-70b-instruct:free` (default)
- `qwen/qwen-2.5-72b-instruct:free`
- `mistralai/mistral-small-3.1-24b-instruct:free`

Free models can be rate-limited or rotate over time — if one stops working, swap
in another. See the current list at <https://openrouter.ai/models?max_price=0>.

---

## Error handling
- **Missing API key** → the app detects it and tells you to set
  `OPENROUTER_API_KEY`, instead of crashing.
- **No notes uploaded / nothing relevant found** → tools answer honestly rather
  than making things up.
- **Bad / unreadable upload** → reported per-file in the ingestion status box.

---

## Deploying to Hugging Face Spaces
This repo ships a **Docker Space**. There's no model server to install — the LLM
is remote (OpenRouter) and the embedding model is baked into the image at build
time — so the container just runs the Gradio app.

### Steps
1. Create a new Space at <https://huggingface.co/new-space>.
2. Set **SDK = Docker** (choose *Blank* / bring-your-own-Dockerfile). The free
   CPU tier is fine.
3. Add your key as a **secret**: Space **Settings → Variables and secrets → New
   secret**, name `OPENROUTER_API_KEY`, value your key. *(Optionally add
   `LLM_MODEL` as a variable to override the model.)*
4. Push this repo to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   *(Or create the Space from the web UI and upload these files.)* The Space
   builds the `Dockerfile` and runs `python app.py`.

### What to expect
- **Build downloads the embedding model** into the image (once, at build time).
- **The first *request* can be slow** — free-tier OpenRouter models may queue —
  so the app shows a brief "heads-up" banner until the first response is served.
- **Storage is ephemeral.** Chroma persists to a writable dir inside the
  container (`CHROMA_DIR`), which resets when the Space restarts. That's fine for
  a demo — re-ingest your notes after a restart. The included `sample_notes.txt`
  lets you try it immediately.

---

## Config knobs (`config.py`)
| Setting | Default | Meaning |
|---------|---------|---------|
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | OpenRouter reasoning + generation model (must support tools) |
| `OPENROUTER_API_KEY` | *(required)* | your OpenRouter key |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embedding model |
| `TOP_K` | `4` | chunks retrieved per query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | splitter settings |
| `CHROMA_DIR` | `./chroma_db` | where the vector store persists |
