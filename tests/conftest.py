"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_image():
    """Provide a sample image for testing."""
    # Mock image data would go here
    return None


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return "A doctor examining a patient"


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    class MockModel:
        def predict(self, x):
            return [0.5] * len(x)
    
    return MockModel()