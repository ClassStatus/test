# Use official Python base image
FROM python:3.11

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ghostscript \
        openjdk-17-jre-headless \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Upgrade pip + install dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install PaddlePaddle and PaddleX OCR explicitly
RUN pip install --no-cache-dir paddlepaddle paddlex[ocr]

# Optional: Pre-download paddlex OCR pipeline model to speed up app startup
RUN python -c "from paddlex import create_pipeline; create_pipeline(pipeline='table_recognition')"

# Copy entire app code
COPY . .

# Create a temporary directory if needed by the app
RUN mkdir -p temp_files

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
