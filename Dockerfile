# Multi-stage build for efficient container size and security
FROM python:3.13-slim as builder

# Set security-focused labels
LABEL maintainer="daniel@terragon.ai" \
      version="0.1.0" \
      security.scan="enabled" \
      org.opencontainers.image.source="https://github.com/terragon-labs/multimodal-counterfactual-lab"

# Install system dependencies for building with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install package and dependencies
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.13-slim as production

# Install runtime system dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with restricted permissions
RUN useradd --create-home --shell /bin/bash --uid 10001 app \
    && mkdir -p /home/app/.cache \
    && chown -R app:app /home/app

USER app
WORKDIR /home/app

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy installed package from builder
COPY --from=builder --chown=app:app /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=app:app /usr/local/bin /usr/local/bin
COPY --from=builder --chown=app:app /app/src ./src

# Expose default Streamlit port (non-privileged)
EXPOSE 8501

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Default command with security considerations
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]