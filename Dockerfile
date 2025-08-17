# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better for Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Start FastAPI app with uvicorn
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]

