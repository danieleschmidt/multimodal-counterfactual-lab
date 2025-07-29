"""Integration tests for counterfactual generation pipeline."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from counterfactual_lab import CounterfactualGenerator


class TestGenerationPipeline:
    """Test complete generation pipeline integration."""
    
    @pytest.fixture
    def mock_image(self):
        """Mock image for testing."""
        mock_img = Mock()
        mock_img.size = (512, 512)
        return mock_img
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "A doctor examining a patient"
    
    @pytest.mark.integration
    def test_modicf_pipeline_integration(self, mock_image, sample_text):
        """Test MoDiCF pipeline end-to-end."""
        with patch('counterfactual_lab.methods.MoDiCF') as mock_modicf:
            # Setup mock
            mock_generator = Mock()
            mock_modicf.return_value = mock_generator
            mock_generator.generate.return_value = [mock_image] * 3
            
            # Test pipeline
            generator = CounterfactualGenerator(method="modicf")
            results = generator.generate(
                image=mock_image,
                text=sample_text,
                attributes=["gender", "age"],
                num_samples=3
            )
            
            # Assertions
            assert len(results) == 3
            mock_generator.generate.assert_called_once()
    
    @pytest.mark.integration
    def test_icg_pipeline_integration(self, mock_image, sample_text):
        """Test ICG pipeline end-to-end."""
        with patch('counterfactual_lab.methods.ICG') as mock_icg:
            # Setup mock
            mock_generator = Mock()
            mock_icg.return_value = mock_generator
            mock_result = Mock()
            mock_result.images = [mock_image] * 2
            mock_result.explanation = "Changed gender from male to female"
            mock_generator.generate_interpretable.return_value = mock_result
            
            # Test pipeline
            generator = CounterfactualGenerator(method="icg")
            results = generator.generate(
                image=mock_image,
                text=sample_text,
                attributes=["gender"],
                num_samples=2
            )
            
            # Assertions
            assert len(results.images) == 2
            assert "gender" in results.explanation
    
    @pytest.mark.integration
    def test_multi_attribute_generation(self, mock_image, sample_text):
        """Test generation with multiple attributes."""
        with patch('counterfactual_lab.methods.MoDiCF') as mock_modicf:
            mock_generator = Mock()
            mock_modicf.return_value = mock_generator
            mock_generator.generate_controlled.return_value = [mock_image] * 5
            
            generator = CounterfactualGenerator(method="modicf")
            results = generator.generate(
                image=mock_image,
                text=sample_text,
                attributes=["gender", "age", "race"],
                num_samples=5
            )
            
            assert len(results) == 5
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_batch_generation(self, mock_image, sample_text):
        """Test batch generation performance."""
        with patch('counterfactual_lab.methods.MoDiCF') as mock_modicf:
            mock_generator = Mock()
            mock_modicf.return_value = mock_generator
            mock_generator.generate_batch.return_value = [[mock_image] * 3] * 10
            
            generator = CounterfactualGenerator(method="modicf")
            
            # Generate batch
            batch_inputs = [(mock_image, sample_text)] * 10
            results = generator.generate_batch(
                batch_inputs,
                attributes=["gender"],
                num_samples=3
            )
            
            assert len(results) == 10
            assert all(len(result) == 3 for result in results)
    
    @pytest.mark.integration
    def test_error_handling_in_pipeline(self, mock_image, sample_text):
        """Test error handling in generation pipeline."""
        with patch('counterfactual_lab.methods.MoDiCF') as mock_modicf:
            # Setup mock to raise exception
            mock_generator = Mock()
            mock_modicf.return_value = mock_generator
            mock_generator.generate.side_effect = RuntimeError("Generation failed")
            
            generator = CounterfactualGenerator(method="modicf")
            
            with pytest.raises(RuntimeError, match="Generation failed"):
                generator.generate(
                    image=mock_image,
                    text=sample_text,
                    attributes=["gender"]
                )