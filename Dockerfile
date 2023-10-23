# Dockerfile

# Use the official Python image with GPU support
FROM python:3.9


# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set the working directory
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# Copy the application
COPY app /app
RUN  apt-get update; \
     apt-get -y upgrade; \
     apt-get -y install wget git python3 python3-venv libgl1 libglib2.0-0;

# Install FastAPI, uvicorn, and required dependencies


# Expose FastAPI's port
EXPOSE 8000

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
