# First stage: Base image with CUDA
FROM  nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        gcc \
        g++ \
        procps \
        sqlite3 \
        python3-pip \
        wkhtmltopdf

        

 RUN echo "1"
# Set the working directory inside the container
WORKDIR /app

# Install llama-cpp-python (build with CUDA)
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"


RUN pip install --upgrade pip && \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Second stage: Python image
#FROM python:3.8-slim-buster as final
#FROM base as final
# Set the working directory inside the container
#WORKDIR /app
#RUN echo "2"
# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY . .

# Expose port 5000 outside of the container
EXPOSE 5000

# Run the application
CMD ["python3", "appv4.py"]

