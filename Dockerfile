# AuraDoc OCR Containerfile with NVIDIA CUDA 12 Support
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements with CUDA support
COPY requirements.txt requirements-cuda.txt ./
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements-cuda.txt

# Copy application source
COPY . .

# Expose Web port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/hardware || exit 1

# Start Uvicorn
CMD ["python3", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
