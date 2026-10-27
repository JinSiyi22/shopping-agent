FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY demo/requirements.txt ./demo/requirements.txt

RUN python -m pip install \
    -r demo/requirements.txt

COPY demo ./demo
COPY data ./data

EXPOSE 8501

CMD ["streamlit", "run", "demo/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
