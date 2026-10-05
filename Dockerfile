FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV PGIM_REPO_ROOT=/app/data \
    PGIM_STATIC_DIR=/app/static \
    PYTHONUNBUFFERED=1 \
    PORT=7860
EXPOSE 7860
# Hosts such as Render pass the port in $PORT.
CMD uvicorn app:site --host 0.0.0.0 --port $PORT
ENV PGIM_DATASETS=R260903,S260903
