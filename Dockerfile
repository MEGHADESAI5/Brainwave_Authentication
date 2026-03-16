# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the entire project
COPY . .

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app.main:app"]