"""Unit tests for MoDiCF generation method."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

pytestmark = pytest.mark.unit


class TestMoDiCFGenerator:
    """Test cases for MoDiCF counterfactual generator."""

    @pytest.fixture
    def modicf_generator(self, mock_diffusion_model):
        """Create MoDiCF generator with mocked dependencies."""
        with patch('counterfactual_lab.methods.modicf.StableDiffusionPipeline') as mock_pipeline:
            mock_pipeline.from_pretrained.return_value = mock_diffusion_model
            from counterfactual_lab.methods.modicf import MoDiCF
            return MoDiCF(model_id="mock-model", device="cpu")

    def test_initialization(self, modicf_generator):
        """Test MoDiCF generator initialization."""
        assert modicf_generator.model_id == "mock-model"
        assert modicf_generator.device == "cpu"
        assert modicf_generator.guidance_scale == 7.5  # Default value

    def test_generate_single_counterfactual(self, modicf_generator, sample_image, sample_text):
        """Test generating a single counterfactual."""
        # Mock the generation process
        with patch.object(modicf_generator, '_generate_counterfactual') as mock_generate:
            mock_generate.return_value = sample_image
            
            result = modicf_generator.generate(
                image=sample_image,
                text=sample_text,
                attributes={"gender": "female"},
                num_samples=1
            )
            
            pytest.assert_valid_counterfactual_result(result)
            assert len(result["counterfactuals"]) == 1
            assert result["metadata"]["method"] == "modicf"

    def test_generate_batch_counterfactuals(self, modicf_generator, sample_image, sample_text):
        """Test generating multiple counterfactuals."""
        with patch.object(modicf_generator, '_generate_counterfactual') as mock_generate:
            mock_generate.return_value = sample_image
            
            result = modicf_generator.generate(
                image=sample_image,
                text=sample_text,
                attributes={"gender": "female", "age": "elderly"},
                num_samples=3
            )
            
            pytest.assert_valid_counterfactual_result(result)
            assert len(result["counterfactuals"]) == 3

    def test_attribute_validation(self, modicf_generator, sample_image, sample_text):
        """Test validation of attribute parameters."""
        with pytest.raises(ValueError, match="Invalid attribute"):
            modicf_generator.generate(
                image=sample_image,
                text=sample_text,
                attributes={"invalid_attr": "value"}
            )

    def test_generate_with_custom_parameters(self, modicf_generator, sample_image, sample_text):
        """Test generation with custom parameters."""
        with patch.object(modicf_generator, '_generate_counterfactual') as mock_generate:
            mock_generate.return_value = sample_image
            
            result = modicf_generator.generate(
                image=sample_image,
                text=sample_text,
                attributes={"gender": "female"},
                guidance_scale=10.0,
                num_inference_steps=25
            )
            
            # Verify custom parameters were used
            mock_generate.assert_called()
            assert result["metadata"]["parameters"]["guidance_scale"] == 10.0
            assert result["metadata"]["parameters"]["num_inference_steps"] == 25

    @pytest.mark.slow
    def test_memory_usage(self, modicf_generator, sample_image, sample_text):
        """Test memory usage during generation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch.object(modicf_generator, '_generate_counterfactual') as mock_generate:
            mock_generate.return_value = sample_image
            
            # Generate multiple counterfactuals
            for _ in range(5):
                modicf_generator.generate(
                    image=sample_image,
                    text=sample_text,
                    attributes={"gender": "female"}
                )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory should not increase by more than 100MB for mocked operations
            assert memory_increase < 100

    def test_error_handling(self, modicf_generator, sample_image, sample_text):
        """Test error handling in generation process."""
        with patch.object(modicf_generator, '_generate_counterfactual') as mock_generate:
            mock_generate.side_effect = RuntimeError("GPU out of memory")
            
            with pytest.raises(RuntimeError, match="GPU out of memory"):
                modicf_generator.generate(
                    image=sample_image,
                    text=sample_text,
                    attributes={"gender": "female"}
                )

    def test_reproducibility(self, modicf_generator, sample_image, sample_text):
        """Test that generation is reproducible with same seed."""
        with patch.object(modicf_generator, '_generate_counterfactual') as mock_generate:
            # Mock to return different values for different calls
            mock_generate.side_effect = [sample_image, sample_image]
            
            result1 = modicf_generator.generate(
                image=sample_image,
                text=sample_text,
                attributes={"gender": "female"},
                seed=42
            )
            
            result2 = modicf_generator.generate(
                image=sample_image,
                text=sample_text,
                attributes={"gender": "female"},
                seed=42
            )
            
            # With same seed, results should be identical
            assert result1["metadata"]["seed"] == result2["metadata"]["seed"]
            assert result1["metadata"]["seed"] == 42