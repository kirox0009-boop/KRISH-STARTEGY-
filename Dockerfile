# KRISH - single image, runs the agent squad and the control room together.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    KRISH_CONFIG_DIR=/app/config \
    KRISH_DATA_DIR=/app/var

WORKDIR /app

# build tools only where needed; kept out of the final layer set as much as slim allows
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/krish/__init__.py backend/krish/__init__.py
RUN pip install --upgrade pip && pip install -e "backend[postgres]"

COPY backend backend
COPY config config
RUN pip install -e "backend[postgres]"

# explicit paths, not brace expansion: RUN uses /bin/sh, which is dash here
RUN mkdir -p /app/var/cache /app/var/artifacts /app/var/packages /app/var/logs
VOLUME ["/app/var"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/overview > /dev/null || exit 1

CMD ["python", "-m", "krish.main", "run"]
