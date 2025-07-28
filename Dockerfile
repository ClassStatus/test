FROM python:3.11

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir paddlepaddle paddlex[ocr]

# Pre-download paddlex OCR models (optional, but speeds up first request)
RUN python -c "from paddlex import create_pipeline; create_pipeline(pipeline='table_recognition')"

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p temp_files

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"] 