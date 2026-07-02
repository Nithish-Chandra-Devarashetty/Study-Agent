# Dockerfile — Hugging Face Docker Space.
#
# The LLM runs remotely on OpenRouter, and embeddings run locally via
# sentence-transformers, so there is NO model server to install or run inside
# the container anymore — just the Python app.
#
# Required Space secret: OPENROUTER_API_KEY  (Settings → Variables and secrets)
FROM python:3.11-slim

# Hugging Face Spaces run the container as a non-root user with uid 1000.
# Create that user and give it a writable HOME (the HF hub caches the embedding
# model under $HOME/.cache, and Chroma persists under the app dir below).
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    CHROMA_DIR=/home/user/app/chroma_db

WORKDIR /home/user/app

# Install Python deps first so this layer is cached across code changes.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# WORKDIR created /home/user/app as ROOT, so hand the whole app dir (and the
# cache dir) to 'user' before we drop privileges — otherwise the non-root user
# can't write chroma_db/ or the model cache at build/runtime.
RUN mkdir -p "$CHROMA_DIR" "$HF_HOME" \
    && chown -R user:user /home/user/app /home/user/.cache

USER user

# Pre-download the embedding model into the image so the first ingest is fast
# and works even if the HF hub is briefly unreachable at runtime.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# The model is now baked into the image, so the app never needs the HF Hub at
# runtime. Staying offline avoids the "set a HF_TOKEN" rate-limit warning. NOTE:
# this MUST come after the download above, or the download itself would fail.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Copy the app code. .dockerignore keeps the local .venv/ and chroma_db/ out.
COPY --chown=user:user . .

# HF Spaces expects the app on port 7860.
EXPOSE 7860

CMD ["python", "app.py"]
