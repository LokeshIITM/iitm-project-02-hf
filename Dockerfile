# syntax=docker/dockerfile:1
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 7860
# IMPORTANT: do NOT set ENV PORT here. HF injects PORT at runtime.
CMD ["bash","-lc","python -m uvicorn api.index:app --host 0.0.0.0 --port $PORT"]
