# Multi-stage build for RAG Chatbot
# Includes automatic model download and dependency installation
# Exposes port 7860 for web interface access

# Use Python 3.8 or higher as the base image
FROM python:3.8-slim

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

RUN ls -lrt /app

# Create required directories
RUN mkdir -p /app/model /app/docs /app/my_vector_db

RUN ls -lrt /app

# Download the model file directly to the correct path
RUN wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf -O /app/model/phi-2.Q4_K_M.gguf

# Copy the requirements file first to leverage caching
COPY requirements.txt .

# Upgrade pip and configure timeout/retry settings
RUN pip install --upgrade pip && \
    pip config set global.timeout 1000 && \
    pip config set global.retries 5

# Install cmake separately with extended timeout
RUN pip install --no-cache-dir cmake --timeout 1000

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

RUN ls -lrt /app

# Expose the port used by the chatbot UI
EXPOSE 7860

# Set the default command to run the RAG chatbot
CMD ["python", "app/rag_chatbot.py"]