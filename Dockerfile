# Multi-stage build for efficient container size
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install package and dependencies
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy installed package from builder
COPY --from=builder --chown=app:app /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=app:app /usr/local/bin /usr/local/bin
COPY --from=builder --chown=app:app /app/src ./src

# Set environment variables
ENV PYTHONPATH=/home/app/src
ENV PYTHONUNBUFFERED=1

# Expose default Streamlit port
EXPOSE 8501

# Default command
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]