# syntax=docker/dockerfile:1.9
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Dependencies in their own layer: editing source does not reinstall spaCy.
# --locked fails the build if uv.lock is stale, which is the failure you want
# at build time rather than at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-editable

# The spaCy model is ~560 MB and changes far less often than the code.
ARG SPACY_MODEL=en_core_web_lg
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run --no-sync python -m spacy download ${SPACY_MODEL}

FROM python:3.12-slim

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data ./data
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser
EXPOSE 8000

# Generous start period: engines load during startup, not on first request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# --factory because there is no module-level app instance.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
