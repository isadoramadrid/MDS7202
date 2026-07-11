FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY frontend/pyproject.toml ./pyproject.toml
COPY frontend/uv.lock ./uv.lock

RUN uv sync --frozen --no-cache

COPY frontend ./frontend

EXPOSE 7860

CMD ["uv", "run", "python", "frontend/app.py"]