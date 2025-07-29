"""Unit tests for core functionality."""

import pytest
from counterfactual_lab.core import CounterfactualGenerator, BiasEvaluator


class TestCounterfactualGenerator:
    """Test counterfactual generator functionality."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = CounterfactualGenerator(method="modicf", device="cpu")
        assert generator.method == "modicf"
        assert generator.device == "cpu"
    
    def test_generate_not_implemented(self):
        """Test that generate method raises NotImplementedError."""
        generator = CounterfactualGenerator()
        with pytest.raises(NotImplementedError):
            generator.generate(None, "test", ["gender"], 1)


class TestBiasEvaluator:
    """Test bias evaluator functionality."""
    
    def test_init(self, mock_model):
        """Test evaluator initialization."""
        evaluator = BiasEvaluator(mock_model)
        assert evaluator.model is mock_model
    
    def test_evaluate_not_implemented(self, mock_model):
        """Test that evaluate method raises NotImplementedError."""
        evaluator = BiasEvaluator(mock_model)
        with pytest.raises(NotImplementedError):
            evaluator.evaluate({}, ["demographic_parity"])