# ─────────────────────────────
# Base image
# ─────────────────────────────
FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# System deps (audio/math libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────
# App setup
# ─────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# ─────────────────────────────
# Expose FastAPI port
# ─────────────────────────────
EXPOSE 8000

# ─────────────────────────────
# Start server
# ─────────────────────────────
CMD ["uvicorn", "live_identity:app", "--host", "0.0.0.0", "--port", "8000"]