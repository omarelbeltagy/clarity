#================================== Description ========================================
# Dockerfile to build the clarity-models service which provides a variety of classification
# models through a FastAPI server.
#=======================================================================================

FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents to the container at /app
COPY app.py .
COPY logging.yaml .
COPY models.yaml .
COPY models/ ./models/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Run the server when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]