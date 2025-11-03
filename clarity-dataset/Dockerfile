#================================== Description ========================================
# Dockerfile to build the clarity-dataset service which is responsible for
# downloading the dataset and saving it in the desired format.
# Supports configurable text cleaning and BERT-based summary generation.

# NOTE:
# The build context should be set to the root directory of the repository when building this
# Docker image, so that the Dockerfile can access the required files and directories.
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
COPY clarity-dataset/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole content of clarity-dataset directory to the container at /app
COPY clarity-dataset/ ./

# Copy the required data to the container
COPY assets/prompts/lora.yaml /app/lora-prompt.yaml

# Set environment variables
ENV DATA_DIR=/app/data
ENV LORA_PROMPT_PATH=/app/lora-prompt.yaml

# Run the script when the container launches
CMD ["python",  "app.py"]