# 📖 Study Companion

A **local LangChain agent** that answers questions, generates quizzes, and
explains concepts — grounded in *your own* study notes. Everything runs on your
machine against [Ollama](https://ollama.com); no OpenAI or external LLM APIs.

The key idea: this is an **agent, not a fixed RAG chain**. The LLM (Qwen2.5:3b)
decides *which tool* to call for each message, and the UI shows you that
decision live.

---

## Architecture

```
you ──▶ Gradio UI (app.py)
             │  your message
             ▼
     Tool-calling agent (agent.py)  ◀── Qwen2.5:3b routes to ONE tool
             │
   ┌─────────┼──────────────┐
   ▼         ▼              ▼
answer_    make_quiz    explain_
from_notes             concept
   │         │              │
   └────── retrieve() ──────┘        (each tool does its own RAG)
             │
             ▼
     Chroma vector store (ingest.py)  ◀── nomic-embed-text embeddings
             │
             ▼
     your uploaded .pdf / .txt notes
```

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
| `config.py` | all settings: model names, paths, `k`, chunk size/overlap |
| `ingest.py` | load → split → embed → persist Chroma; retrieval & de-dup |
| `agent.py` | the three tools + the tool-calling agent loop |
| `app.py` | Gradio UI (upload, chat, live agent-step display) |

---

## Setup

### 1. Install Python dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull the Ollama models
Make sure Ollama is installed and running (`ollama serve`), then:
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

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

## Error handling
- **Ollama not running** → the app detects it and tells you to `ollama serve` +
  pull the models, instead of crashing.
- **No notes uploaded / nothing relevant found** → tools answer honestly rather
  than making things up.
- **Bad / unreadable upload** → reported per-file in the ingestion status box.

---

## Deploying to Hugging Face Spaces
This repo ships a **Docker Space** (not the Gradio SDK) that runs **Ollama and
both models entirely inside the container** — no external model backend. It's
built for a demo on the **free CPU tier**: it prioritises "stands up and works"
over speed.

**What's in the box:**
| File | Role |
|------|------|
| `Dockerfile` | slim Python base → installs Ollama → installs deps → runs `start.sh` |
| `start.sh` | starts `ollama serve`, waits until it's reachable, pulls the models, launches the app |
| `.dockerignore` | keeps the local `.venv/` and `chroma_db/` out of the image |

### Steps
1. Create a new Space at <https://huggingface.co/new-space>.
2. Set **SDK = Docker** (choose *Blank* / bring-your-own-Dockerfile), and pick
   the **free CPU** hardware.
3. Push this repo to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   *(Or create the Space from the web UI and upload these files.)* The Space
   builds the `Dockerfile` and runs `start.sh` as its entrypoint.
4. Watch the build/run **Logs** tab — `start.sh` prints each step (`1/4 …` →
   `4/4 Launching …`) so you can follow the startup and model downloads.

### What to expect (CPU-tier realities)
- **First startup is slow.** On first boot the container downloads the model
  weights into itself — **~2GB for `qwen2.5:3b`** plus the `nomic-embed-text`
  embedding model — before the app comes up. This is normal; the logs show the
  pull progress.
- **The first *request* is slow too.** The app shows a **"⏳ warming up"** banner
  until the first response is served, because Ollama has to load the model into
  memory on CPU. Later requests are faster. There's no GPU anywhere — this runs
  on the free CPU tier by design.
- **Free Spaces sleep after inactivity.** When the Space wakes, that cold first
  request is slow again while the model reloads.
- **Storage is ephemeral.** Chroma persists to a writable dir inside the
  container (`CHROMA_DIR`), which resets when the Space restarts. That's fine
  for a demo — just re-ingest your notes after a restart. The included
  `sample_notes.txt` lets you try it immediately.

> The LangChain agent, its three tools, and the model choices are **identical**
> to the local version — only this deployment wrapper is added.

---

## Config knobs (`config.py`)
| Setting | Default | Meaning |
|---------|---------|---------|
| `LLM_MODEL` | `qwen2.5:3b` | reasoning + generation model |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `TOP_K` | `4` | chunks retrieved per query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | splitter settings |
| `CHROMA_DIR` | `./chroma_db` | where the vector store persists |
