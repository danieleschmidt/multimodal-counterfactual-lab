"""Performance tests for memory usage."""

import pytest
import psutil
import time
from unittest.mock import Mock, patch

from counterfactual_lab import CounterfactualGenerator, BiasEvaluator


class TestMemoryUsage:
    """Test memory usage patterns."""
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.performance
    @patch('counterfactual_lab.methods.MoDiCF')
    def test_generation_memory_usage(self, mock_modicf):
        """Test memory usage during generation."""
        # Setup mock
        mock_generator_obj = Mock()
        mock_modicf.return_value = mock_generator_obj
        mock_generator_obj.generate.return_value = [Mock()] * 10
        
        memory_before = self.get_memory_usage()
        
        # Create generator and run generation
        generator = CounterfactualGenerator(method="modicf")
        results = generator.generate(
            image=Mock(),
            text="Test text",
            attributes=["gender", "age"],
            num_samples=10
        )
        
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        # Memory usage should be reasonable (less than 500MB for mock)
        assert memory_used < 500, f"Memory usage too high: {memory_used:.1f}MB"
        assert len(results) == 10
    
    @pytest.mark.performance
    @patch('counterfactual_lab.BiasEvaluator')
    def test_evaluation_memory_scaling(self, mock_evaluator_class):
        """Test memory scaling for evaluation."""
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.evaluate.return_value = {"score": 0.5}
        
        evaluator = BiasEvaluator(Mock())
        
        # Test different dataset sizes
        memory_usage = {}
        dataset_sizes = [10, 50, 100]
        
        for size in dataset_sizes:
            memory_before = self.get_memory_usage()
            
            # Simulate evaluation with dataset of given size
            mock_dataset = [Mock() for _ in range(size)]
            evaluator.evaluate(
                counterfactuals=mock_dataset,
                metrics=["demographic_parity"]
            )
            
            memory_after = self.get_memory_usage()
            memory_usage[size] = memory_after - memory_before
        
        # Memory should scale reasonably with dataset size
        # (allowing for some variance in mock testing)
        assert memory_usage[100] >= memory_usage[10]
        assert memory_usage[100] < memory_usage[10] * 20  # Should not scale too badly
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        initial_memory = self.get_memory_usage()
        
        with patch('counterfactual_lab.CounterfactualGenerator') as mock_gen_class:
            mock_generator = Mock()
            mock_gen_class.return_value = mock_generator
            mock_generator.generate.return_value = [Mock()]
            
            # Perform many operations
            for i in range(20):
                generator = CounterfactualGenerator(method="modicf")
                generator.generate(
                    image=Mock(),
                    text=f"Test text {i}",
                    attributes=["gender"]
                )
                
                # Force garbage collection periodically
                if i % 5 == 0:
                    import gc
                    gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 100MB for mocks)
        assert memory_increase < 100, f"Potential memory leak: {memory_increase:.1f}MB increase"
    
    @pytest.mark.performance
    def test_concurrent_memory_usage(self):
        """Test memory usage under concurrent operations."""
        import threading
        import queue
        
        memory_readings = queue.Queue()
        
        def worker():
            """Worker function for concurrent testing."""
            with patch('counterfactual_lab.CounterfactualGenerator') as mock_gen_class:
                mock_generator = Mock()
                mock_gen_class.return_value = mock_generator
                mock_generator.generate.return_value = [Mock()]
                
                generator = CounterfactualGenerator(method="icg")
                generator.generate(Mock(), "Test", ["gender"])
                
                # Record memory usage
                memory_readings.put(self.get_memory_usage())
        
        initial_memory = self.get_memory_usage()
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check peak memory usage during concurrent operations
        peak_memory = max(list(memory_readings.queue))
        memory_increase = peak_memory - initial_memory
        
        # Concurrent memory usage should be reasonable
        assert memory_increase < 200, f"Concurrent memory usage too high: {memory_increase:.1f}MB"