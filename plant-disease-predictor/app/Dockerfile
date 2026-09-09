FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for Pillow / image processing if slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose Streamlit port
EXPOSE 80

# Configure healthcheck
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]
