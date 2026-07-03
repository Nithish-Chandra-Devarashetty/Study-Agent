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
*which tool* to call for each message — including a real judgment call between
**"answer from the notes"** and **"the notes don't cover this, go search the
web"** — and the UI shows you that decision live.

---

## Architecture

```
you ──▶ Gradio UI (app.py)
             │  your message
             ▼
     Tool-calling agent (agent.py)  ◀── OpenRouter LLM routes to a tool
             │
   ┌─────────┼───────────┬──────────────┐
   ▼         ▼           ▼               ▼
answer_    make_quiz  explain_       search_web
from_notes            concept        (FALLBACK)
   │         │           │               │
   └────── retrieve() ───┘          DuckDuckGo (no key)
             │                           │
             ▼                           ▼
     Chroma vector store          the public web
     (local ST embeddings)
             │
             ▼
     your uploaded .pdf / .txt notes
```

**The agentic decision.** `answer_from_notes` is always tried first. If retrieval
comes back empty it returns a `NOTES_MISS` signal; the agent then *decides*
whether to fall back to `search_web` (for general-knowledge questions) or to tell
you honestly that it isn't in your notes. Web-sourced answers are clearly
labelled as **not** grounded in your notes. Set `ENABLE_WEB_SEARCH=0` to drop the
tool entirely and keep the agent strictly notes-only.

**Why two different backends?** OpenRouter is a chat-completions service — it has
no embeddings endpoint. So the LLM comes from OpenRouter, while embeddings run
in-process with a small `sentence-transformers` model (free, no server, no key).

### The four tools
| Tool | When the agent uses it | What it returns |
|------|------------------------|-----------------|
| `answer_from_notes` | factual questions ("What is X?") | grounded answer + source snippet |
| `make_quiz` | "quiz me on…", "test me" | N mixed MCQ + short-answer questions **with an answer key** |
| `explain_concept` | "explain… simply", confusion | plain-language explanation + one analogy |
| `search_web` | **fallback** — only after the notes miss, for general knowledge | web answer, clearly labelled *not from your notes* + source URL |

The first three retrieve the top-k relevant chunks from Chroma and generate their
answer **only** from those chunks, so responses stay grounded in your notes.
`search_web` is the one door out of the notes, taken only when the agent decides
they don't cover the question (uses DuckDuckGo — free, no API key).

### Files
| File | Role |
|------|------|
| `config.py` | all settings: model names, OpenRouter key/URL, paths, `k`, chunk size, web-search toggle |
| `ingest.py` | load → split → embed (local) → persist Chroma; retrieval & de-dup |
| `websearch.py` | DuckDuckGo web-search helper for the `search_web` fallback tool |
| `agent.py` | the four tools + the tool-calling agent loop (OpenRouter LLM) |
| `app.py` | Gradio UI (upload, chat, live agent-step display) |
| `evaluate.py` | routing-accuracy + answer-groundedness evaluation harness |
| `eval/` | fixed notes corpus + labelled test set the evaluation runs against |

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
   - *"What's the capital of Australia?"* (not in your notes) → the agent tries
     the notes, sees a miss, and falls back to `search_web`
3. Watch the **🛠 Agent steps & sources** panel to see which tool the agent
   chose and the source snippet it grounded on.

Re-uploading the same file **replaces** its chunks (no duplicates); uploading a
new file **adds** to the collection. Use **🧹 Clear all notes** to start fresh.

---

## Evaluation

A RAG *agent* has two things worth measuring, so there's a small, reproducible
harness for exactly those:

1. **Routing accuracy** — for each message, did the agent pick the *right* tool?
   This includes the hard case: did it fall back to `search_web` when (and only
   when) the notes don't contain the answer?
2. **Answer groundedness** — did the answer stay supported by the retrieved
   notes instead of drifting into invented facts? Graded by an **LLM-as-judge**
   that only ever sees the retrieved notes, the question, and the answer, and
   returns one label (`GROUNDED` / `PARTIAL` / `UNSUPPORTED` / `REFUSED`).

It runs against a **fixed corpus** (`eval/eval_notes.txt`) and an **18-case
labelled test set** (`eval/eval_set.json`) — 8 factual, 3 quiz, 3 explain, and 4
deliberately out-of-notes questions — so scores are reproducible run to run. The
eval ingests into a throwaway vector store, so your real `./chroma_db` is left
untouched.

```bash
python evaluate.py                 # full run → prints metrics, writes eval/results.md
python evaluate.py --no-judge      # routing only (skips the LLM judge calls)
python evaluate.py --limit 6       # quick smoke test on the first 6 cases
```

The report breaks down three numbers: **routing accuracy** (all cases),
**groundedness** (in-notes cases), and **honest web fallback** (out-of-notes
cases went to the web or refused — never fabricated a notes-based answer). A
per-case table lands in `eval/results.md`.

---

## Metrics & limitations

Latest `python evaluate.py` run (18 cases, model `poolside/laguna-m.1`; full
per-case breakdown in [`eval/results.md`](eval/results.md)):

| Metric | Score | What it means |
|--------|-------|---------------|
| **Routing accuracy** | **89%** (16/18) | agent picked the right tool, including the web fallback |
| **Groundedness** (in-notes) | **100%** (14/14) | in-notes answers were supported by the retrieved notes (LLM judge) |
| **Honest web fallback** | **100%** (4/4) | every out-of-notes question went to the web or refused — none was fabricated from the notes |

**What works well**
- **Grounding is the strong point — 100%.** Every in-notes answer stayed
  supported by the retrieved chunks; each tool answers only from what it
  retrieved and cites the source snippet.
- **The agentic decision holds up — 100% honest fallback.** All four
  out-of-notes questions were tried against the notes first, correctly detected
  as misses, and routed to `search_web` (clearly labelled as web-sourced). None
  was answered from thin air.
- **Intent routing is reliable** for quiz/explain phrasings ("quiz me…",
  "explain… simply") — those 6 cases routed perfectly.

**Where it slips (the 2 routing misses)**
- Both failures are *over-eager extra tool calls*, not wrong answers: on one
  water-cycle question the agent answered from notes **and then also** ran a web
  search (needless), and on one "explain like I'm five" it called
  `explain_concept` then re-queried `answer_from_notes`. The final answers were
  still grounded — the metric penalises the messy trajectory, which is the honest
  thing to measure.

**What doesn't (known limitations)**
- **Retrieval is plain top-k cosine** over `all-MiniLM-L6-v2` (384-dim). Fine for
  clean prose; it can miss paraphrased or multi-hop questions, and there's no
  re-ranking or query rewriting.
- **Free-tier LLM variance.** Small/free OpenRouter models occasionally emit a
  malformed tool call or over-eagerly reach for `search_web`. `handle_parsing_errors`
  and argument-coercion validators soften this, but routing isn't 100%.
- **Web fallback is best-effort.** DuckDuckGo results can be thin or rate-limited;
  the tool summarises only snippets (it doesn't fetch full pages).
- **Groundedness is judged by an LLM**, not humans — a reasonable proxy, but the
  judge itself can be wrong on borderline cases.
- **Ephemeral storage on HF Spaces** — the vector store resets on restart; re-ingest
  after a restart.

**What I'd do next**
- Add a **retrieval-quality** metric (hit@k against gold chunk IDs) so answer
  errors can be attributed to retrieval vs. generation.
- **Hybrid retrieval** (BM25 + dense) and a cross-encoder re-ranker for recall.
- **Confidence-gated web fallback** driven by a retrieval-score threshold rather
  than an empty-result check, plus full-page fetch + citation of web sources.
- Grow the eval set and add **inter-rater checks** on the LLM judge (multiple
  judges / a small human-labelled slice) to calibrate the groundedness number.

---

## Choosing a model
Set `LLM_MODEL` (env var or `.env`) to any **tool-calling-capable** OpenRouter
model. This is a tool-calling agent, so the model *must* support function calling.
The default is `poolside/laguna-m.1`; other good free options:
- `meta-llama/llama-3.3-70b-instruct:free`
- `qwen/qwen-2.5-72b-instruct:free`
- `mistralai/mistral-small-3.1-24b-instruct:free`

Free models can be rate-limited or rotate over time — if one stops working, swap
in another. See the current list at <https://openrouter.ai/models?max_price=0>.

---

## Error handling
- **Missing API key** → the app detects it and tells you to set
  `OPENROUTER_API_KEY`, instead of crashing.
- **No notes uploaded / nothing relevant found** → the agent either falls back to
  a web search (labelled as such) or answers honestly rather than making things up.
- **Web search unavailable** (offline / rate-limited) → `search_web` returns an
  honest "couldn't reach the web" instead of crashing the agent loop.
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
| `LLM_MODEL` | `poolside/laguna-m.1` | OpenRouter reasoning + generation model (must support tools) |
| `OPENROUTER_API_KEY` | *(required)* | your OpenRouter key |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embedding model |
| `ENABLE_WEB_SEARCH` | `1` | offer the `search_web` fallback tool (`0` = notes-only) |
| `WEB_SEARCH_RESULTS` | `4` | web results summarised per fallback search |
| `TOP_K` | `4` | chunks retrieved per query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | splitter settings |
| `CHROMA_DIR` | `./chroma_db` | where the vector store persists |
