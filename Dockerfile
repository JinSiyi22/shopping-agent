FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# sentence-transformers requires PyTorch.
# Install the CPU-only build so the runtime image does not pull CUDA/NVIDIA packages.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.1

COPY requirements-runtime.txt .

RUN python -m pip install \
    -r requirements-runtime.txt

COPY app ./app
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
