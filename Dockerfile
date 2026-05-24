# ──────────────────────────────────────────────
# Stage 1: builder
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libpq-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2: runner
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runner
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder --chown=appuser:appgroup /install /usr/local
COPY --chown=appuser:appgroup . .
RUN mkdir -p /cache/fastembed && chown -R appuser:appgroup /cache/fastembed
USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
