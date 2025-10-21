#================================== Description ========================================
# Dockerfile to build the clarity-dataset service which is responsible for
# downloading the dataset and save it in the desired format.
#=======================================================================================

FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents to the container at /app
COPY . .

# Run the script when the container launches
CMD ["python",  "app.py"]