# Use an official lightweight Python image.
FROM python:3.8-slim

# Prevents Python from writing .pyc files to disc and buffers stdout and stderr.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt /app/requirements.txt

# Install dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container.
COPY . /app

# Expose port 80 (Render/Cloud Run will set the port via the $PORT environment variable).
EXPOSE 80

# Start the FastAPI application using Uvicorn.
<<<<<<< HEAD
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
=======
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
>>>>>>> 5dcd9a7c81658c39fadffeda585410790b9a296b
