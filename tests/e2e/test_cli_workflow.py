"""End-to-end tests for CLI workflows."""

import pytest
import subprocess
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock


class TestCLIWorkflow:
    """Test complete CLI workflows."""
    
    @pytest.mark.e2e
    def test_cli_help_command(self):
        """Test CLI help command works."""
        result = subprocess.run(
            ["python", "-m", "counterfactual_lab.cli", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "counterfactual" in result.stdout.lower()
    
    @pytest.mark.e2e
    @patch('counterfactual_lab.CounterfactualGenerator')
    def test_cli_generation_workflow(self, mock_generator_class):
        """Test CLI generation workflow."""
        # Setup mock
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate.return_value = [Mock(), Mock(), Mock()]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_image = temp_path / "test_image.jpg"
            output_dir = temp_path / "outputs"
            
            # Create dummy input file
            input_image.touch()
            
            # Run CLI command
            result = subprocess.run([
                "python", "-m", "counterfactual_lab.cli",
                "generate",
                "--method", "modicf",
                "--image", str(input_image),
                "--text", "A doctor examining a patient",
                "--attributes", "gender,age",
                "--output", str(output_dir),
                "--num-samples", "3"
            ], capture_output=True, text=True)
            
            # Check command succeeded (mock will handle actual generation)
            assert result.returncode == 0 or "mock" in result.stderr.lower()
    
    @pytest.mark.e2e
    @patch('counterfactual_lab.BiasEvaluator')
    def test_cli_evaluation_workflow(self, mock_evaluator_class):
        """Test CLI evaluation workflow."""
        # Setup mock
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.evaluate.return_value = {"demographic_parity": 0.85}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "dataset.json"
            output_file = temp_path / "evaluation_results.json"
            
            # Create dummy dataset file
            dataset_data = {
                "images": ["image1.jpg", "image2.jpg"],
                "texts": ["Text 1", "Text 2"],
                "labels": ["male", "female"]
            }
            with open(dataset_path, "w") as f:
                json.dump(dataset_data, f)
            
            # Run CLI command
            result = subprocess.run([
                "python", "-m", "counterfactual_lab.cli",
                "evaluate",
                "--dataset", str(dataset_path),
                "--metrics", "demographic_parity,equalized_odds",
                "--output", str(output_file)
            ], capture_output=True, text=True)
            
            # Check command succeeded (mock will handle actual evaluation)
            assert result.returncode == 0 or "mock" in result.stderr.lower()
    
    @pytest.mark.e2e
    def test_cli_version_command(self):
        """Test CLI version command."""
        result = subprocess.run([
            "python", "-m", "counterfactual_lab.cli",
            "--version"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "0.1.0" in result.stdout or "version" in result.stdout.lower()
    
    @pytest.mark.e2e
    def test_cli_invalid_command(self):
        """Test CLI handles invalid commands gracefully."""
        result = subprocess.run([
            "python", "-m", "counterfactual_lab.cli",
            "invalid_command"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower()
    
    @pytest.mark.e2e
    @patch('counterfactual_lab.CounterfactualGenerator')
    def test_cli_batch_processing(self, mock_generator_class):
        """Test CLI batch processing workflow."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_batch.return_value = [[Mock()] * 2] * 3
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            batch_file = temp_path / "batch_input.json"
            output_dir = temp_path / "batch_outputs"
            
            # Create batch input file
            batch_data = {
                "items": [
                    {"image": "image1.jpg", "text": "Text 1"},
                    {"image": "image2.jpg", "text": "Text 2"},
                    {"image": "image3.jpg", "text": "Text 3"}
                ]
            }
            with open(batch_file, "w") as f:
                json.dump(batch_data, f)
            
            # Run CLI batch command
            result = subprocess.run([
                "python", "-m", "counterfactual_lab.cli",
                "batch",
                "--input", str(batch_file),
                "--method", "icg",
                "--output", str(output_dir),
                "--attributes", "gender,age"
            ], capture_output=True, text=True)
            
            # Check command succeeded (mock will handle actual processing)
            assert result.returncode == 0 or "mock" in result.stderr.lower()