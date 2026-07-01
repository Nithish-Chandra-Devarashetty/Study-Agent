#!/usr/bin/env bash
#
# start.sh — container entrypoint for the Hugging Face Docker Space.
#
# Ollama runs INSIDE the container, so before the Gradio app can talk to it we
# must (a) start the server, (b) wait until its HTTP API is actually reachable
# — this is the startup race guard — and only then (c) pull the models and
# (d) launch the app. Every step logs so the Space build/run log is readable.
#
set -e

echo "[start] Booting Study Companion (Hugging Face Docker Space)…"

# 1. Start the Ollama server in the background. This returns immediately; the
#    server is NOT listening yet, which is exactly the race we guard against.
echo "[start] 1/4 Starting Ollama server in the background…"
ollama serve &
OLLAMA_PID=$!

# 2. Block until the Ollama API answers HTTP 200. Without this gate, the model
#    pull below (and the app) could fire before the server binds :11434 and hit
#    "connection refused", crash-looping the container on boot.
echo "[start] 2/4 Waiting for the Ollama API to become reachable…"
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
  # Fail fast (instead of looping forever) if the server process died.
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[start] ERROR: Ollama exited before it was ready." >&2
    exit 1
  fi
  echo "[start]   …not up yet, retrying in 1s"
  sleep 1
done
echo "[start] Ollama API is up."

# 3. Pull the models only if they aren't already cached. On the very first boot
#    this downloads ~2GB for qwen2.5:3b plus the embedding model, so it's slow.
echo "[start] 3/4 Ensuring models are present…"
for MODEL in qwen2.5:3b nomic-embed-text; do
  if ollama list | grep -q "$MODEL"; then
    echo "[start]   '$MODEL' already present — skipping pull."
  else
    echo "[start]   Pulling '$MODEL' (first boot downloads weights, be patient)…"
    ollama pull "$MODEL"
    echo "[start]   Done pulling '$MODEL'."
  fi
done

# 4. Launch the Gradio app in the foreground (keeps the container alive). By now
#    Ollama is confirmed up and the models are present, so there's no race.
echo "[start] 4/4 Launching the Gradio app on 0.0.0.0:7860…"
exec python app.py
