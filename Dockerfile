FROM python:3.9-slim

# Ensure apt works properly by setting environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies with retry logic to prevent transient network issues
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app
WORKDIR /app

# Expose the necessary port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app4.py", "--server.port=8501", "--server.enableCORS=false"]
