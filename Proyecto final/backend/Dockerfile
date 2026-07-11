FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY backend/pyproject.toml ./pyproject.toml
COPY backend/uv.lock ./uv.lock

RUN uv sync --frozen --no-cache

COPY backend ./backend

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]