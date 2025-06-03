# Pull from light Python:3.8 base Docker image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
