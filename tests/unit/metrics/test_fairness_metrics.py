"""Unit tests for fairness metrics."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

pytestmark = pytest.mark.unit


class TestFairnessMetrics:
    """Test cases for fairness evaluation metrics."""

    @pytest.fixture
    def fairness_evaluator(self):
        """Create fairness evaluator instance."""
        from counterfactual_lab.metrics.fairness import FairnessEvaluator
        return FairnessEvaluator()

    @pytest.fixture
    def mock_predictions(self):
        """Mock model predictions for testing."""
        return {
            "positive_predictions": np.array([1, 0, 1, 0, 1, 1, 0, 0]),
            "protected_attributes": np.array([0, 0, 0, 0, 1, 1, 1, 1]),  # 0=group A, 1=group B
            "true_labels": np.array([1, 0, 1, 0, 1, 0, 1, 0])
        }

    def test_demographic_parity(self, fairness_evaluator, mock_predictions):
        """Test demographic parity metric calculation."""
        dp_score = fairness_evaluator.demographic_parity(
            predictions=mock_predictions["positive_predictions"],
            protected_attributes=mock_predictions["protected_attributes"]
        )
        
        # Demographic parity should be between 0 and 1
        assert 0 <= dp_score <= 1
        assert isinstance(dp_score, float)

    def test_equal_opportunity(self, fairness_evaluator, mock_predictions):
        """Test equal opportunity metric calculation."""
        eo_score = fairness_evaluator.equal_opportunity(
            predictions=mock_predictions["positive_predictions"],
            true_labels=mock_predictions["true_labels"],
            protected_attributes=mock_predictions["protected_attributes"]
        )
        
        assert 0 <= eo_score <= 1
        assert isinstance(eo_score, float)

    def test_disparate_impact(self, fairness_evaluator, mock_predictions):
        """Test disparate impact metric calculation."""
        di_score = fairness_evaluator.disparate_impact(
            predictions=mock_predictions["positive_predictions"],
            protected_attributes=mock_predictions["protected_attributes"]
        )
        
        # Disparate impact can be > 1, but should be positive
        assert di_score > 0
        assert isinstance(di_score, float)

    def test_statistical_parity_distance(self, fairness_evaluator, mock_predictions):
        """Test statistical parity distance metric."""
        spd_score = fairness_evaluator.statistical_parity_distance(
            predictions=mock_predictions["positive_predictions"],
            protected_attributes=mock_predictions["protected_attributes"]
        )
        
        # SPD should be between -1 and 1
        assert -1 <= spd_score <= 1
        assert isinstance(spd_score, float)

    def test_average_odds_difference(self, fairness_evaluator, mock_predictions):
        """Test average odds difference metric."""
        aod_score = fairness_evaluator.average_odds_difference(
            predictions=mock_predictions["positive_predictions"],
            true_labels=mock_predictions["true_labels"],
            protected_attributes=mock_predictions["protected_attributes"]
        )
        
        assert -1 <= aod_score <= 1
        assert isinstance(aod_score, float)

    def test_evaluate_all_metrics(self, fairness_evaluator, mock_predictions):
        """Test evaluating all fairness metrics at once."""
        results = fairness_evaluator.evaluate_all(
            predictions=mock_predictions["positive_predictions"],
            true_labels=mock_predictions["true_labels"],
            protected_attributes=mock_predictions["protected_attributes"]
        )
        
        pytest.assert_valid_evaluation_result(results)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "demographic_parity",
            "equal_opportunity", 
            "disparate_impact",
            "statistical_parity_distance",
            "average_odds_difference"
        ]
        
        for metric in expected_metrics:
            assert metric in results

    def test_empty_input_handling(self, fairness_evaluator):
        """Test handling of empty inputs."""
        with pytest.raises(ValueError, match="Empty input"):
            fairness_evaluator.demographic_parity(
                predictions=np.array([]),
                protected_attributes=np.array([])
            )

    def test_mismatched_input_lengths(self, fairness_evaluator):
        """Test handling of mismatched input lengths."""
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            fairness_evaluator.demographic_parity(
                predictions=np.array([1, 0, 1]),
                protected_attributes=np.array([0, 1])  # Different length
            )

    def test_single_group_handling(self, fairness_evaluator):
        """Test handling when only one protected group is present."""
        # All samples from same group
        predictions = np.array([1, 0, 1, 0])
        protected_attributes = np.array([0, 0, 0, 0])  # All same group
        
        # Should handle gracefully, possibly returning neutral score
        dp_score = fairness_evaluator.demographic_parity(
            predictions=predictions,
            protected_attributes=protected_attributes
        )
        
        # When only one group, demographic parity should be 1.0 (perfect)
        assert dp_score == 1.0

    def test_bias_threshold_checking(self, fairness_evaluator, mock_predictions):
        """Test bias threshold checking functionality."""
        results = fairness_evaluator.evaluate_all(
            predictions=mock_predictions["positive_predictions"],
            true_labels=mock_predictions["true_labels"],
            protected_attributes=mock_predictions["protected_attributes"]
        )
        
        # Test bias detection with different thresholds
        bias_detected_strict = fairness_evaluator.detect_bias(results, threshold=0.9)
        bias_detected_lenient = fairness_evaluator.detect_bias(results, threshold=0.5)
        
        assert isinstance(bias_detected_strict, bool)
        assert isinstance(bias_detected_lenient, bool)

    def test_multiclass_protected_attributes(self, fairness_evaluator):
        """Test metrics with multiple protected attribute groups."""
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
        protected_attributes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # 3 groups
        
        dp_score = fairness_evaluator.demographic_parity(
            predictions=predictions,
            protected_attributes=protected_attributes
        )
        
        assert 0 <= dp_score <= 1
        assert isinstance(dp_score, float)

    @pytest.mark.performance
    def test_large_scale_evaluation(self, fairness_evaluator):
        """Test performance with large datasets."""
        # Generate large mock dataset
        n_samples = 10000
        predictions = np.random.binomial(1, 0.5, n_samples)
        protected_attributes = np.random.binomial(1, 0.3, n_samples)
        true_labels = np.random.binomial(1, 0.4, n_samples)
        
        import time
        start_time = time.time()
        
        results = fairness_evaluator.evaluate_all(
            predictions=predictions,
            true_labels=true_labels,
            protected_attributes=protected_attributes
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (< 1 second for 10k samples)
        assert execution_time < 1.0
        pytest.assert_valid_evaluation_result(results)