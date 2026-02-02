FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ backend/
COPY frontend/ frontend/
COPY presentation/ presentation/
COPY Bitcoin_Price_Prediction_Report.pdf .

# Create directory for saved models (if mapped volume not used, though volume is recommended)
RUN mkdir -p backend/saved_models

# Expose port
EXPOSE 8000

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Environment variable to control service type (default to api)
ENV SERVICE_TYPE=api

# Command to run the application via entrypoint
ENTRYPOINT ["./entrypoint.sh"]
