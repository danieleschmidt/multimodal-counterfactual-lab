#!/bin/bash

# Multimodal Counterfactual Lab Development Environment Setup

set -e

echo "🚀 Setting up Multimodal Counterfactual Lab development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    tree \
    jq \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx

# Upgrade pip and install wheel
echo "🐍 Upgrading pip and installing wheel..."
python -m pip install --upgrade pip wheel setuptools

# Install pre-commit
echo "🔧 Installing pre-commit..."
python -m pip install pre-commit

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models data outputs cache logs results
mkdir -p models/transformers models/huggingface models/torch
mkdir -p tests/data tests/fixtures

# Install Python dependencies
echo "📚 Installing Python dependencies..."
if [ -f "requirements-dev.txt" ]; then
    python -m pip install -r requirements-dev.txt
fi

if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
fi

# Install the package in development mode
echo "🔨 Installing package in development mode..."
python -m pip install -e .

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Setup git configuration for development
echo "🔧 Configuring git..."
git config --global --add safe.directory /workspace

# Create .env file from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📄 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your configuration values"
fi

# Download small test models if in development mode
if [ "${COUNTERFACTUAL_LAB_ENV:-development}" = "development" ]; then
    echo "🤗 Downloading development models..."
    python -c "
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/models/transformers'
os.environ['HF_HOME'] = '/workspace/models/huggingface'

try:
    from transformers import AutoTokenizer, AutoModel
    # Download small models for development
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    print('✅ Downloaded DistilBERT for development')
except Exception as e:
    print(f'⚠️  Could not download models: {e}')
"
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
if [ -f "pytest.ini" ] || [ -f "pyproject.toml" ]; then
    python -m pytest tests/ -v --tb=short || echo "⚠️  Some tests failed - this is normal during setup"
fi

# Check code quality tools
echo "🔍 Checking code quality tools..."
python -m ruff --version || echo "⚠️  Ruff not available"
python -m black --version || echo "⚠️  Black not available"
python -m mypy --version || echo "⚠️  MyPy not available"

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Add your HuggingFace token to .env (HUGGINGFACE_TOKEN)"
echo "3. Run 'make test' to verify everything works"
echo "4. Run 'streamlit run src/counterfactual_lab/app.py' to start the web interface"
echo "5. Check the README.md for detailed usage instructions"
echo ""
echo "🚀 Happy coding!"