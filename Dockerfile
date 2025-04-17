FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
    poppler-utils \
    mupdf \
    unzip \
    zip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p filein uploads processed/OCR processed/failed processed/processed_originals temp_splits

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]