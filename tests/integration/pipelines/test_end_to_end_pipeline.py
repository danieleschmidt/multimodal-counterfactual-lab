"""Integration tests for end-to-end counterfactual generation and evaluation pipeline."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

pytestmark = pytest.mark.integration


class TestEndToEndPipeline:
    """Test complete pipeline from input to final evaluation."""

    @pytest.fixture
    def pipeline_config(self):
        """Configuration for pipeline testing."""
        return {
            "generation": {
                "method": "modicf",
                "batch_size": 2,
                "num_samples": 3,
                "attributes": ["gender", "race"]
            },
            "evaluation": {
                "metrics": ["demographic_parity", "cits_score"],
                "thresholds": {"fairness": 0.8, "quality": 0.85}
            },
            "output": {
                "save_images": True,
                "save_reports": True,
                "format": "json"
            }
        }

    @pytest.fixture
    def mock_pipeline_components(self):
        """Mock all pipeline components for integration testing."""
        with patch('counterfactual_lab.CounterfactualGenerator') as mock_generator, \
             patch('counterfactual_lab.BiasEvaluator') as mock_evaluator, \
             patch('counterfactual_lab.Pipeline') as mock_pipeline:
            
            # Configure mock generator
            mock_gen_instance = Mock()
            mock_gen_instance.generate.return_value = {
                "counterfactuals": ["mock_image1", "mock_image2", "mock_image3"],
                "metadata": {"method": "modicf", "parameters": {}}
            }
            mock_generator.return_value = mock_gen_instance
            
            # Configure mock evaluator
            mock_eval_instance = Mock()
            mock_eval_instance.evaluate.return_value = {
                "demographic_parity": 0.85,
                "cits_score": 0.91
            }
            mock_evaluator.return_value = mock_eval_instance
            
            # Configure mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            yield {
                "generator": mock_gen_instance,
                "evaluator": mock_eval_instance,
                "pipeline": mock_pipeline_instance
            }

    def test_complete_pipeline_execution(self, pipeline_config, mock_pipeline_components, 
                                       sample_image, sample_text, temp_dir):
        """Test complete pipeline from generation to evaluation."""
        from counterfactual_lab import Pipeline
        
        # Create pipeline instance
        pipeline = Pipeline(config=pipeline_config)
        
        # Execute pipeline
        result = pipeline.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir
        )
        
        # Verify pipeline execution
        assert "generation_results" in result
        assert "evaluation_results" in result
        assert "metadata" in result
        
        # Verify generation was called
        mock_pipeline_components["generator"].generate.assert_called_once()
        
        # Verify evaluation was called
        mock_pipeline_components["evaluator"].evaluate.assert_called_once()

    def test_batch_pipeline_execution(self, pipeline_config, mock_pipeline_components,
                                    sample_image_batch, sample_text_batch, temp_dir):
        """Test pipeline execution with batch inputs."""
        from counterfactual_lab import Pipeline
        
        pipeline = Pipeline(config=pipeline_config)
        
        # Process batch
        results = pipeline.run_batch(
            images=sample_image_batch,
            texts=sample_text_batch,
            output_dir=temp_dir
        )
        
        # Should have results for each input pair
        assert len(results) == len(sample_image_batch)
        
        # Each result should have the expected structure
        for result in results:
            assert "generation_results" in result
            assert "evaluation_results" in result

    def test_pipeline_with_file_outputs(self, pipeline_config, mock_pipeline_components,
                                      sample_image, sample_text, temp_dir):
        """Test pipeline with file output generation."""
        from counterfactual_lab import Pipeline
        
        pipeline_config["output"]["save_images"] = True
        pipeline_config["output"]["save_reports"] = True
        
        pipeline = Pipeline(config=pipeline_config)
        
        result = pipeline.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir
        )
        
        # Check that output files would be created (mocked)
        assert "output_paths" in result
        assert result["output_paths"]["images"] is not None
        assert result["output_paths"]["report"] is not None

    def test_pipeline_error_recovery(self, pipeline_config, mock_pipeline_components,
                                   sample_image, sample_text, temp_dir):
        """Test pipeline error handling and recovery."""
        from counterfactual_lab import Pipeline
        
        # Make generator fail initially
        mock_pipeline_components["generator"].generate.side_effect = [
            RuntimeError("Generation failed"),
            {  # Successful retry
                "counterfactuals": ["mock_image1"],
                "metadata": {"method": "modicf", "retry": True}
            }
        ]
        
        pipeline = Pipeline(config=pipeline_config, max_retries=2)
        
        # Should recover from initial failure
        result = pipeline.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir
        )
        
        # Verify retry occurred
        assert mock_pipeline_components["generator"].generate.call_count == 2
        assert result["metadata"]["retries"] > 0

    def test_pipeline_quality_filtering(self, pipeline_config, mock_pipeline_components,
                                      sample_image, sample_text, temp_dir):
        """Test pipeline quality filtering of generated counterfactuals."""
        from counterfactual_lab import Pipeline
        
        # Configure generator to return mixed quality results
        mock_pipeline_components["generator"].generate.return_value = {
            "counterfactuals": ["high_quality", "low_quality", "medium_quality"],
            "quality_scores": [0.95, 0.65, 0.82],
            "metadata": {"method": "modicf"}
        }
        
        # Configure evaluator to filter based on quality
        mock_pipeline_components["evaluator"].filter_by_quality.return_value = {
            "counterfactuals": ["high_quality", "medium_quality"],
            "quality_scores": [0.95, 0.82]
        }
        
        pipeline_config["evaluation"]["quality_threshold"] = 0.8
        pipeline = Pipeline(config=pipeline_config)
        
        result = pipeline.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir
        )
        
        # Verify quality filtering was applied
        filtered_results = result["generation_results"]["counterfactuals"]
        assert len(filtered_results) == 2  # Low quality filtered out

    def test_pipeline_fairness_validation(self, pipeline_config, mock_pipeline_components,
                                        sample_image, sample_text, temp_dir):
        """Test pipeline fairness validation and reporting."""
        from counterfactual_lab import Pipeline
        
        # Configure evaluator to return fairness metrics
        mock_pipeline_components["evaluator"].evaluate.return_value = {
            "demographic_parity": 0.75,  # Below threshold
            "cits_score": 0.91
        }
        
        pipeline_config["evaluation"]["thresholds"]["fairness"] = 0.8
        pipeline = Pipeline(config=pipeline_config)
        
        result = pipeline.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir
        )
        
        # Check fairness validation results
        assert "fairness_validation" in result
        assert result["fairness_validation"]["passed"] is False
        assert "demographic_parity" in result["fairness_validation"]["failed_metrics"]

    def test_pipeline_memory_management(self, pipeline_config, mock_pipeline_components,
                                      sample_image_batch, sample_text_batch, temp_dir):
        """Test pipeline memory management with large batches."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        from counterfactual_lab import Pipeline
        
        # Configure for memory-efficient processing
        pipeline_config["generation"]["batch_size"] = 1  # Process one at a time
        pipeline_config["memory"]["max_usage_mb"] = 1024
        
        pipeline = Pipeline(config=pipeline_config)
        
        # Process large batch
        results = pipeline.run_batch(
            images=sample_image_batch * 10,  # Larger batch
            texts=sample_text_batch * 10,
            output_dir=temp_dir
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be controlled
        assert memory_increase < 500  # Less than 500MB increase
        assert len(results) == len(sample_image_batch) * 10

    @pytest.mark.slow
    def test_pipeline_performance_benchmarking(self, pipeline_config, mock_pipeline_components,
                                             sample_image, sample_text, temp_dir):
        """Test pipeline performance characteristics."""
        from counterfactual_lab import Pipeline
        import time
        
        pipeline = Pipeline(config=pipeline_config)
        
        # Measure execution time
        start_time = time.time()
        
        result = pipeline.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify performance metrics are recorded
        assert "performance" in result["metadata"]
        assert result["metadata"]["performance"]["execution_time"] > 0
        
        # For mocked operations, should be very fast
        assert execution_time < 1.0

    def test_pipeline_configuration_validation(self):
        """Test pipeline configuration validation."""
        from counterfactual_lab import Pipeline
        
        # Test invalid configuration
        invalid_config = {
            "generation": {"method": "invalid_method"},
            "evaluation": {"metrics": ["invalid_metric"]}
        }
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            Pipeline(config=invalid_config)

    def test_pipeline_reproducibility(self, pipeline_config, mock_pipeline_components,
                                    sample_image, sample_text, temp_dir):
        """Test pipeline reproducibility with seed control."""
        from counterfactual_lab import Pipeline
        
        pipeline_config["seed"] = 42
        
        pipeline1 = Pipeline(config=pipeline_config)
        pipeline2 = Pipeline(config=pipeline_config)
        
        result1 = pipeline1.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir / "run1"
        )
        
        result2 = pipeline2.run(
            image=sample_image,
            text=sample_text,
            output_dir=temp_dir / "run2"
        )
        
        # Results should be reproducible with same seed
        assert result1["metadata"]["seed"] == result2["metadata"]["seed"]
        assert result1["metadata"]["seed"] == 42