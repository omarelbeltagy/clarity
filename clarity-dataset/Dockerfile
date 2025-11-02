#================================== Description ========================================
# Dockerfile to build the clarity-dataset service which is responsible for
# downloading the dataset and saving it in the desired format.
# Supports configurable text cleaning and BERT-based summary generation.
#=======================================================================================

FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents to the container at /app
COPY app.py .
COPY clean.py .
COPY summarize.py .
COPY logging.yaml .

# Set entrypoint to python app.py so flags can be passed
ENTRYPOINT ["python", "app.py"]