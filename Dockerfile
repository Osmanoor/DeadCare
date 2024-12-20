FROM python:3.9-slim

# Install Tesseract and language packs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 


# Install required Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the app code
COPY . /app
WORKDIR /app

# Run the Streamlit app
CMD ["streamlit", "run", "app4.py", "--server.port=8501"]
