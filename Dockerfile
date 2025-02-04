# Use specific Python version as base image
FROM python:3.10.11-slim

# Set working directory
WORKDIR /app

# Copy repository contents
COPY . /app

# Copy .env file (ensure it's in your .dockerignore)
COPY .env /app/.env

# Install system dependencies and Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose application port
EXPOSE 8000

# Command to run the application
CMD ["python", "application.py"]
```

Build and run commands:
```bash
# Build docker image
docker build -t pid-backend .

# Run docker container
docker run -p 8000:8000 pid-backend
```

Key Considerations:
- Ensure .env is in .dockerignore
- Use specific port mapping
- Verify requirements.txt is accurate