FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Download the 2-bit quantized model (Phi-2)
RUN wget -O models/phi-2.Q2_K.gguf \
    https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K.gguf

# Copy application code
COPY app.py .

# Expose the port the app runs on
EXPOSE 5020

# Command to run the application
CMD ["python", "app.py"]
