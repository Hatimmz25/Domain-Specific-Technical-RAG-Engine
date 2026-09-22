# ==============================================================================
# Stage 1: Builder Stage (Compiles C++ binaries & builds python wheels)
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build tools required for compiling C++ extensions (FAISS, llama-cpp-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into wheels cache
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# ==============================================================================
# Stage 2: Final Runtime Stage (Clean, minimal, non-root)
# ==============================================================================
FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Install minimal runtime system requirements (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and persistent storage directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/raw /app/data/processed /app/indexes /app/models && \
    chown -R appuser:appuser /app

WORKDIR /app

# Switch to non-root user
USER appuser

# Copy built wheels from builder and install into user path
COPY --from=builder --chown=appuser:appuser /build/wheels /tmp/wheels
RUN pip install --no-cache-dir --user /tmp/wheels/* && rm -rf /tmp/wheels

# Copy application source code into image
COPY --chown=appuser:appuser . .

# Expose FastAPI backend (8000) and Streamlit frontend (8501) ports
EXPOSE 8000
EXPOSE 8501

# Default command runs FastAPI backend server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]