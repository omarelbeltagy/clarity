#================================== Description ========================================
# Dockerfile to build the clarity-models service which provides a variety of classification
# models through a FastAPI server.
#
# NOTE:
# The build context should be set to the root directory of the repository when building this
# Docker image, so that the Dockerfile can access the required files and directories.
#=======================================================================================

FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt
COPY clarity-models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole content of clarity-models directory to the container at /app
COPY clarity-models/ ./

# Copy the training data to the container at /app/data
COPY data/simple/ ./app/data/simple/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Run the server when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]