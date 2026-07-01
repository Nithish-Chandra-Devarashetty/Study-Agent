"""
app.py — Gradio entry point (the UI).

Layout:
  - Left column: upload .pdf / .txt notes, ingest them, or clear the collection.
  - Main column: a chat interface + a live "Agent steps & sources" panel that
    surfaces WHICH tool the agent chose and the source snippet it grounded on.

The chat handler is a *generator*: it streams the agent's progress so the user
sees "Thinking…", then "🔍 Retrieving from notes…", then the final answer —
making the agent's routing decisions visible, exactly as the brief requires.

Run:  python app.py   (then open the printed local URL)
"""

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

import config
from ingest import ingest_files, clear_all, check_ollama, collection_count
from agent import build_agent, TOOL_STATUS

# Build the agent once at startup. Constructing it does NOT contact Ollama —
# the connection only happens when we actually invoke it during a chat.
agent_executor = build_agent()

# On CPU (the free Spaces tier), the very first request is slow because Ollama
# has to load qwen2.5:3b into memory. We show a "warming up" banner until the
# first response has actually been served, then hide it. Module-level flag so it
# stays hidden for every later request.
_first_response_served = False


# --- Ingestion callbacks --------------------------------------------------

def handle_ingest(files):
    """Called by the 'Ingest notes' button. `files` is a list of temp paths."""
    if not files:
        return "Please select at least one .pdf or .txt file first."
    # Gradio gives file objects/paths depending on version; normalise to paths.
    paths = [f.name if hasattr(f, "name") else f for f in files]
    return ingest_files(paths)


def handle_clear():
    """Called by the 'Clear all notes' button."""
    return clear_all()


# --- Chat callbacks -------------------------------------------------------

def add_user_message(message, history):
    """Append the user's message and clear the input box."""
    if not message.strip():
        return "", history
    history = history + [{"role": "user", "content": message}]
    return "", history


def _to_lc_history(history):
    """Convert prior Gradio messages into LangChain messages for the agent.

    We pass everything EXCEPT the final user turn (that goes in as `input`).
    """
    lc = []
    for turn in history[:-1]:
        if turn["role"] == "user":
            lc.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            lc.append(AIMessage(content=turn["content"]))
    return lc


def _extract_source(observation: str) -> str:
    """Pull the 'SOURCE: ...' line a tool appended to its result, if present."""
    if not isinstance(observation, str) or "SOURCE:" not in observation:
        return ""
    return "SOURCE:" + observation.split("SOURCE:", 1)[1].strip()


def agent_respond(history):
    """Generator: stream the agent loop, yielding UI updates as it goes.

    Yields (chatbot_history, steps_markdown, warmup_banner_update) on each step
    so the user watches the agent think, route to a tool, and answer. The banner
    update is `gr.update()` (no change) until the first response is served, at
    which point we hide the "warming up" notice for good.
    """
    global _first_response_served
    message = history[-1]["content"]
    keep_banner = gr.update()  # leave the warm-up banner as-is on this yield

    # Guard: Ollama must be running for the agent to do anything.
    if not check_ollama():
        err = ("⚠️ **Ollama isn't running.** Start it and pull the models:\n\n"
               "```\nollama serve\nollama pull qwen2.5:3b\nollama pull nomic-embed-text\n```")
        history = history + [{"role": "assistant", "content": err}]
        yield history, "❌ Could not reach Ollama.", keep_banner
        return

    # Soft warning if there are no notes yet (tools also handle this honestly).
    prefix = ""
    if collection_count() == 0:
        prefix = "> ℹ️ No notes are ingested yet — upload some for grounded answers.\n\n"

    steps_log = ["🤔 **Thinking…** deciding which tool to use."]
    yield history, prefix + "\n".join(steps_log), keep_banner

    chat_history = _to_lc_history(history)
    final_answer = ""

    try:
        # AgentExecutor.stream emits the agent loop step-by-step:
        #   - chunks with "actions"  -> the LLM chose a tool (the routing decision)
        #   - chunks with "steps"    -> a tool ran; we grab its observation/source
        #   - a chunk with "output"  -> the final answer
        for chunk in agent_executor.stream(
            {"input": message, "chat_history": chat_history}
        ):
            if "actions" in chunk:
                for action in chunk["actions"]:
                    status = TOOL_STATUS.get(action.tool, f"🔧 Running `{action.tool}`…")
                    steps_log.append(f"**Tool chosen:** `{action.tool}` — {status}")
                    yield history, prefix + "\n\n".join(steps_log), keep_banner

            elif "steps" in chunk:
                for step in chunk["steps"]:
                    source = _extract_source(step.observation)
                    if source:
                        steps_log.append(f"**Grounded on:**\n\n> {source}")
                        yield history, prefix + "\n\n".join(steps_log), keep_banner

            elif "output" in chunk:
                final_answer = chunk["output"]

    except Exception as exc:
        final_answer = (f"Something went wrong talking to the model: `{exc}`\n\n"
                        "Check that Ollama is running and the models are pulled.")

    if not final_answer:
        final_answer = "I couldn't produce an answer this time — please try rephrasing."

    history = history + [{"role": "assistant", "content": final_answer}]
    steps_log.append("✅ **Done.**")

    # First real response is done → retire the "warming up" banner.
    banner_update = keep_banner
    if not _first_response_served:
        _first_response_served = True
        banner_update = gr.update(visible=False)
    yield history, prefix + "\n\n".join(steps_log), banner_update


# --- UI layout ------------------------------------------------------------

with gr.Blocks(title="Study Companion") as demo:
    gr.Markdown(
        "# 📖 Study Companion\n"
        "A local LangChain **agent** that answers, quizzes, and explains — "
        "grounded in *your own* study notes. Ask a question, ask to be quizzed, "
        "or ask for a simple explanation, and watch which tool the agent picks."
    )

    # Warm-up notice: visible until the first response is served. On the free
    # CPU tier, loading qwen2.5:3b into memory makes that first reply slow.
    warmup_banner = gr.Markdown(
        "⏳ **Warming up** — the model is loading into memory. Your **first "
        "response may take a minute or two on CPU**; later ones are faster.",
        visible=True,
    )

    with gr.Row():
        # ---- Left: notes management ----
        with gr.Column(scale=1):
            gr.Markdown("### 1. Add your notes")
            file_input = gr.File(
                label="Upload .pdf / .txt notes",
                file_count="multiple",
                file_types=[".pdf", ".txt"],
            )
            ingest_btn = gr.Button("📥 Ingest notes", variant="primary")
            clear_btn = gr.Button("🧹 Clear all notes")
            ingest_status = gr.Textbox(
                label="Ingestion status", lines=6, interactive=False
            )

        # ---- Right: chat + agent steps ----
        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask your Study Companion")
            # Gradio 6 uses the {"role", "content"} message format by default
            # (the old `type="messages"` arg was removed).
            chatbot = gr.Chatbot(label="Chat", height=420)
            msg = gr.Textbox(
                label="Your message",
                placeholder="e.g. 'Quiz me on photosynthesis' or 'Explain osmosis simply'",
            )
            steps_panel = gr.Markdown(
                "🛠 **Agent steps & sources** will appear here.",
                label="Agent steps",
            )

    # ---- Wiring ----
    ingest_btn.click(handle_ingest, inputs=file_input, outputs=ingest_status)
    clear_btn.click(handle_clear, inputs=None, outputs=ingest_status)

    # Two-stage submit: first show the user's message, then stream the agent.
    # The agent step also drives the warm-up banner (hidden after first reply).
    msg.submit(add_user_message, [msg, chatbot], [msg, chatbot]).then(
        agent_respond, chatbot, [chatbot, steps_panel, warmup_banner]
    )


if __name__ == "__main__":
    # A quick heads-up in the console if Ollama looks down at launch.
    if not check_ollama():
        print("⚠️  Ollama not reachable at", config.OLLAMA_BASE_URL,
              "\n   Start it with `ollama serve` and pull the models "
              "(see README).")
    # Bind to 0.0.0.0:7860 so the app is reachable inside the Docker Space (and
    # this works unchanged locally — just open http://127.0.0.1:7860).
    demo.launch(server_name="0.0.0.0", server_port=7860)
