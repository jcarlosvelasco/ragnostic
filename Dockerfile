# ──────────────────────────────────────────────
# Stage 1: builder
# ──────────────────────────────────────────────
FROM python:3.11-alpine AS builder

WORKDIR /backend

RUN apk add --no-cache gcc musl-dev libffi-dev postgresql-dev

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2: runner
# ──────────────────────────────────────────────
FROM python:3.11-alpine AS runner

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /backend

RUN apk add --no-cache libpq

COPY --from=builder --chown=appuser:appgroup /install /usr/local
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
