# Dockerfile — self-contained Hugging Face Docker Space.
#
# Everything runs INSIDE this one container on the free CPU tier: the Ollama
# server, the qwen2.5:3b + nomic-embed-text models (pulled at startup), and the
# Gradio app. No GPU, no external model backend.
FROM python:3.11-slim

# curl is needed both to install Ollama and for start.sh's health-check poll;
# ca-certificates lets the installer/model download reach the internet over TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the Ollama CPU build (this is the standard, GPU-optional installer).
RUN curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face Spaces run the container as a non-root user with uid 1000.
# Create that user and give it a writable HOME (Ollama stores pulled model
# weights under $HOME/.ollama, and Chroma persists under the app dir below).
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/.ollama/models \
    CHROMA_DIR=/home/user/app/chroma_db

WORKDIR /home/user/app

# Install Python deps first so this layer is cached across code changes. The
# dependency list is identical to the local app — only the wrapper differs.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the startup script. .dockerignore keeps the local
# .venv/ and chroma_db/ out of the image.
COPY --chown=user:user . .
RUN chmod +x start.sh

USER user

# HF Spaces expects the app on port 7860.
EXPOSE 7860

CMD ["./start.sh"]
