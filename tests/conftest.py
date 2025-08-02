"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest
from PIL import Image


# Test configuration
@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure environment variables for testing."""
    os.environ["COUNTERFACTUAL_LAB_ENV"] = "test"
    os.environ["ENABLE_TELEMETRY"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU for tests
    os.environ["TRANSFORMERS_CACHE"] = str(Path(__file__).parent / "fixtures" / "models")


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image():
    """Provide a sample RGB image for testing."""
    # Create a simple 224x224 RGB test image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image_array, mode='RGB')


@pytest.fixture
def sample_image_batch():
    """Provide a batch of sample images for testing."""
    batch = []
    for _ in range(4):
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        batch.append(Image.fromarray(image_array, mode='RGB'))
    return batch


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return "A doctor examining a patient"


@pytest.fixture
def sample_text_batch():
    """Provide a batch of sample texts for testing."""
    return [
        "A doctor examining a patient",
        "A teacher in a classroom",
        "An engineer working on a project",
        "A nurse caring for patients"
    ]


@pytest.fixture
def sample_attributes():
    """Provide sample attribute definitions for testing."""
    return {
        "gender": ["male", "female"],
        "race": ["white", "black", "asian", "hispanic"],
        "age": ["young", "middle", "elderly"]
    }


@pytest.fixture
def sample_counterfactuals():
    """Provide sample counterfactual results for testing."""
    return {
        "original": {
            "image": "mock_image",
            "text": "A young white male doctor examining a patient",
            "attributes": {"gender": "male", "race": "white", "age": "young"}
        },
        "counterfactuals": [
            {
                "image": "mock_image_cf1",
                "text": "A young white female doctor examining a patient",
                "attributes": {"gender": "female", "race": "white", "age": "young"}
            },
            {
                "image": "mock_image_cf2", 
                "text": "A young black male doctor examining a patient",
                "attributes": {"gender": "male", "race": "black", "age": "young"}
            }
        ]
    }


# Model mocks
@pytest.fixture
def mock_diffusion_model():
    """Mock diffusion model for testing."""
    mock = Mock()
    mock.generate.return_value = Mock()  # Mock generated image
    mock.device = "cpu"
    return mock


@pytest.fixture
def mock_text_encoder():
    """Mock text encoder for testing."""
    mock = Mock()
    mock.encode.return_value = np.random.randn(512)  # Mock text embedding
    return mock


@pytest.fixture
def mock_vision_model():
    """Mock vision model for testing."""
    mock = Mock()
    mock.encode_image.return_value = np.random.randn(512)  # Mock image embedding
    mock.encode_text.return_value = np.random.randn(512)   # Mock text embedding
    return mock


@pytest.fixture
def mock_attribute_classifier():
    """Mock attribute classifier for testing."""
    mock = Mock()
    mock.predict.return_value = {
        "gender": {"male": 0.7, "female": 0.3},
        "race": {"white": 0.6, "black": 0.2, "asian": 0.1, "hispanic": 0.1},
        "age": {"young": 0.8, "middle": 0.15, "elderly": 0.05}
    }
    return mock


# Pipeline mocks  
@pytest.fixture
def mock_modicf_generator():
    """Mock MoDiCF generator for testing."""
    mock = Mock()
    mock.generate.return_value = {
        "counterfactuals": ["mock_image1", "mock_image2"],
        "metadata": {"method": "modicf", "parameters": {}}
    }
    return mock


@pytest.fixture
def mock_icg_generator():
    """Mock ICG generator for testing."""
    mock = Mock()
    mock.generate.return_value = {
        "counterfactuals": ["mock_image1", "mock_image2"],
        "explanations": ["Changed gender from male to female"],
        "metadata": {"method": "icg", "parameters": {}}
    }
    return mock


@pytest.fixture
def mock_bias_evaluator():
    """Mock bias evaluator for testing."""
    mock = Mock()
    mock.evaluate.return_value = {
        "demographic_parity": 0.85,
        "equal_opportunity": 0.78,
        "disparate_impact": 0.82,
        "cits_score": 0.91
    }
    return mock


# Configuration fixtures
@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "generation": {
            "method": "modicf",
            "batch_size": 2,
            "num_samples": 4,
            "attributes": ["gender", "race"]
        },
        "evaluation": {
            "metrics": ["demographic_parity", "cits_score"],
            "thresholds": {"fairness": 0.8, "quality": 0.85}
        },
        "models": {
            "diffusion": "stabilityai/stable-diffusion-2-1",
            "vision": "openai/clip-vit-base-patch32"
        }
    }


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    return [
        {
            "image_path": "test_image_1.jpg",
            "text": "A person in a professional setting",
            "attributes": {"gender": "male", "race": "white"}
        },
        {
            "image_path": "test_image_2.jpg", 
            "text": "A person working in an office",
            "attributes": {"gender": "female", "race": "black"}
        }
    ]


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_memory_mb": 2048,
        "max_time_seconds": 30,
        "batch_sizes": [1, 2, 4, 8],
        "image_sizes": [(224, 224), (256, 256), (512, 512)]
    }


# Pytest marks for test organization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu
pytest.mark.mutation = pytest.mark.mutation


# Custom assertions
def assert_valid_counterfactual_result(result):
    """Assert that a counterfactual result has the expected structure."""
    assert "counterfactuals" in result
    assert "metadata" in result
    assert isinstance(result["counterfactuals"], list)
    assert len(result["counterfactuals"]) > 0


def assert_valid_evaluation_result(result):
    """Assert that an evaluation result has the expected structure."""
    assert isinstance(result, dict)
    assert all(isinstance(v, (int, float)) for v in result.values())
    assert all(0 <= v <= 1 for v in result.values())  # Assuming normalized metrics


# Add custom assertions to pytest namespace
pytest.assert_valid_counterfactual_result = assert_valid_counterfactual_result
pytest.assert_valid_evaluation_result = assert_valid_evaluation_result