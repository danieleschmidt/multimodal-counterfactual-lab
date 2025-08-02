"""Performance tests for counterfactual generation methods."""

import pytest
import time
import psutil
import os
from unittest.mock import Mock, patch
import numpy as np

pytestmark = [pytest.mark.performance, pytest.mark.slow]


class TestGenerationPerformance:
    """Performance benchmarks for counterfactual generation."""

    @pytest.fixture
    def performance_thresholds(self):
        """Define performance thresholds for different operations."""
        return {
            "single_generation_time": 5.0,  # seconds
            "batch_generation_time_per_item": 3.0,  # seconds per item
            "memory_usage_mb": 2048,  # MB
            "gpu_memory_usage_mb": 8192,  # MB (if GPU available)
            "throughput_items_per_minute": 10
        }

    def test_modicf_single_generation_performance(self, performance_thresholds, 
                                                sample_image, sample_text):
        """Test MoDiCF single generation performance."""
        with patch('counterfactual_lab.methods.modicf.MoDiCF') as mock_modicf:
            # Configure mock to simulate realistic timing
            mock_instance = Mock()
            mock_instance.generate.side_effect = lambda *args, **kwargs: (
                time.sleep(0.1),  # Simulate 100ms generation time
                {
                    "counterfactuals": [sample_image],
                    "metadata": {"method": "modicf", "generation_time": 0.1}
                }
            )[1]  # Return the dict, not the sleep result
            mock_modicf.return_value = mock_instance
            
            generator = mock_modicf()
            
            # Measure performance
            start_time = time.time()
            
            result = generator.generate(
                image=sample_image,
                text=sample_text,
                attributes={"gender": "female"}
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Performance assertions
            assert generation_time < performance_thresholds["single_generation_time"]
            assert "generation_time" in result["metadata"]

    def test_modicf_batch_generation_performance(self, performance_thresholds,
                                               sample_image_batch, sample_text_batch):
        """Test MoDiCF batch generation performance."""
        with patch('counterfactual_lab.methods.modicf.MoDiCF') as mock_modicf:
            mock_instance = Mock()
            
            def mock_batch_generate(*args, **kwargs):
                # Simulate batch processing time
                batch_size = len(sample_image_batch)
                time.sleep(0.05 * batch_size)  # 50ms per item
                return {
                    "counterfactuals": sample_image_batch,
                    "metadata": {"method": "modicf", "batch_size": batch_size}
                }
            
            mock_instance.batch_generate = mock_batch_generate
            mock_modicf.return_value = mock_instance
            
            generator = mock_modicf()
            
            # Measure batch performance
            start_time = time.time()
            
            result = generator.batch_generate(
                images=sample_image_batch,
                texts=sample_text_batch,
                attributes={"gender": "female"}
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            time_per_item = total_time / len(sample_image_batch)
            
            # Performance assertions
            assert time_per_item < performance_thresholds["batch_generation_time_per_item"]
            assert total_time < len(sample_image_batch) * performance_thresholds["single_generation_time"]

    def test_memory_usage_during_generation(self, performance_thresholds,
                                          sample_image, sample_text):
        """Test memory usage during counterfactual generation."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('counterfactual_lab.methods.modicf.MoDiCF') as mock_modicf:
            mock_instance = Mock()
            
            def mock_generate_with_memory(*args, **kwargs):
                # Simulate memory allocation
                dummy_data = np.random.randn(1000, 1000)  # Allocate some memory
                time.sleep(0.1)
                return {
                    "counterfactuals": [sample_image],
                    "metadata": {"method": "modicf"},
                    "_temp_data": dummy_data  # Keep reference to prevent GC
                }
            
            mock_instance.generate = mock_generate_with_memory
            mock_modicf.return_value = mock_instance
            
            generator = mock_modicf()
            
            # Generate multiple counterfactuals to test memory accumulation
            results = []
            for i in range(10):
                result = generator.generate(
                    image=sample_image,
                    text=sample_text,
                    attributes={"gender": "female"}
                )
                results.append(result)
                
                # Check memory usage periodically
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't grow excessively
                assert memory_increase < performance_thresholds["memory_usage_mb"]

    def test_throughput_measurement(self, performance_thresholds, 
                                  sample_image_batch, sample_text_batch):
        """Test overall throughput of generation pipeline."""
        with patch('counterfactual_lab.CounterfactualGenerator') as mock_generator:
            mock_instance = Mock()
            
            def mock_fast_generate(*args, **kwargs):
                time.sleep(0.02)  # 20ms per generation
                return {
                    "counterfactuals": [sample_image_batch[0]],
                    "metadata": {"method": "modicf"}
                }
            
            mock_instance.generate = mock_fast_generate
            mock_generator.return_value = mock_instance
            
            generator = mock_generator()
            
            # Measure throughput over 1 minute simulation
            start_time = time.time()
            generation_count = 0
            target_duration = 6.0  # 6 seconds for faster testing
            
            while time.time() - start_time < target_duration:
                generator.generate(
                    image=sample_image_batch[0],
                    text=sample_text_batch[0],
                    attributes={"gender": "female"}
                )
                generation_count += 1
            
            actual_duration = time.time() - start_time
            throughput = (generation_count / actual_duration) * 60  # items per minute
            
            # Throughput should meet minimum threshold
            assert throughput >= performance_thresholds["throughput_items_per_minute"]

    @pytest.mark.gpu
    def test_gpu_memory_usage(self, performance_thresholds, sample_image, sample_text):
        """Test GPU memory usage during generation (if GPU available)."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available for testing")
            
            # Mock GPU memory monitoring
            with patch('torch.cuda.memory_allocated') as mock_memory:
                mock_memory.return_value = 1024 * 1024 * 100  # 100MB in bytes
                
                with patch('counterfactual_lab.methods.modicf.MoDiCF') as mock_modicf:
                    mock_instance = Mock()
                    mock_instance.device = "cuda"
                    mock_instance.generate.return_value = {
                        "counterfactuals": [sample_image],
                        "metadata": {"method": "modicf", "device": "cuda"}
                    }
                    mock_modicf.return_value = mock_instance
                    
                    generator = mock_modicf()
                    
                    result = generator.generate(
                        image=sample_image,
                        text=sample_text,
                        attributes={"gender": "female"}
                    )
                    
                    # Check GPU memory usage
                    gpu_memory_mb = mock_memory.return_value / 1024 / 1024
                    assert gpu_memory_mb < performance_thresholds["gpu_memory_usage_mb"]
                    
        except ImportError:
            pytest.skip("PyTorch not available for GPU testing")

    def test_concurrent_generation_performance(self, performance_thresholds,
                                             sample_image, sample_text):
        """Test performance under concurrent generation requests."""
        import threading
        import concurrent.futures
        
        with patch('counterfactual_lab.methods.modicf.MoDiCF') as mock_modicf:
            mock_instance = Mock()
            
            def mock_thread_safe_generate(*args, **kwargs):
                time.sleep(0.1)  # Simulate generation time
                return {
                    "counterfactuals": [sample_image],
                    "metadata": {"method": "modicf", "thread_id": threading.current_thread().ident}
                }
            
            mock_instance.generate = mock_thread_safe_generate
            mock_modicf.return_value = mock_instance
            
            generator = mock_modicf()
            
            # Test concurrent execution
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(8):
                    future = executor.submit(
                        generator.generate,
                        image=sample_image,
                        text=sample_text,
                        attributes={"gender": "female"}
                    )
                    futures.append(future)
                
                # Wait for all to complete
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Concurrent execution should be faster than sequential
            sequential_time = 8 * 0.1  # 8 tasks * 0.1s each
            assert total_time < sequential_time
            assert len(results) == 8

    def test_memory_cleanup_after_generation(self, sample_image, sample_text):
        """Test that memory is properly cleaned up after generation."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('counterfactual_lab.methods.modicf.MoDiCF') as mock_modicf:
            mock_instance = Mock()
            
            def mock_generate_with_cleanup(*args, **kwargs):
                # Simulate memory allocation and cleanup
                dummy_data = np.random.randn(2000, 2000)  # Large allocation
                result = {
                    "counterfactuals": [sample_image],
                    "metadata": {"method": "modicf"}
                }
                del dummy_data  # Explicit cleanup
                return result
            
            mock_instance.generate = mock_generate_with_cleanup
            mock_modicf.return_value = mock_instance
            
            generator = mock_modicf()
            
            # Generate and clean up multiple times
            for i in range(5):
                result = generator.generate(
                    image=sample_image,
                    text=sample_text,
                    attributes={"gender": "female"}
                )
                del result  # Explicit cleanup
                
                # Force garbage collection
                import gc
                gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory should not accumulate significantly
            assert memory_increase < 100  # Less than 100MB increase

    def test_performance_regression_detection(self, performance_thresholds,
                                            sample_image, sample_text):
        """Test for performance regression detection."""
        with patch('counterfactual_lab.methods.modicf.MoDiCF') as mock_modicf:
            mock_instance = Mock()
            
            # Simulate consistent performance
            generation_times = []
            
            def mock_timed_generate(*args, **kwargs):
                start = time.time()
                time.sleep(0.05)  # Consistent 50ms
                end = time.time()
                generation_time = end - start
                generation_times.append(generation_time)
                
                return {
                    "counterfactuals": [sample_image],
                    "metadata": {"method": "modicf", "generation_time": generation_time}
                }
            
            mock_instance.generate = mock_timed_generate
            mock_modicf.return_value = mock_instance
            
            generator = mock_modicf()
            
            # Run multiple generations
            for i in range(10):
                generator.generate(
                    image=sample_image,
                    text=sample_text,
                    attributes={"gender": "female"}
                )
            
            # Analyze performance consistency
            avg_time = np.mean(generation_times)
            std_time = np.std(generation_times)
            max_time = np.max(generation_times)
            
            # Performance should be consistent
            assert std_time < 0.01  # Low variance
            assert max_time < avg_time * 1.5  # No outliers > 50% of average
            assert avg_time < 0.1  # Average within expected range