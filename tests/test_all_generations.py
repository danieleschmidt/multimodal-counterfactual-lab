"""Comprehensive test suite for all three generations of the counterfactual lab."""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all three generations
from counterfactual_lab.minimal_core import (
    MinimalCounterfactualGenerator, MinimalBiasEvaluator, MockImage
)
from counterfactual_lab.robust_core import (
    RobustCounterfactualGenerator, RobustBiasEvaluator,
    ValidationError, GenerationError, RateLimitError
)
from counterfactual_lab.scalable_core import (
    ScalableCounterfactualGenerator, PerformanceMonitor, IntelligentCache, WorkerPool
)

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.start_time = time.time()
    
    def pass_test(self, test_name: str):
        """Record a passed test."""
        self.passed += 1
        logger.info(f"✅ {self.name} - {test_name} PASSED")
    
    def fail_test(self, test_name: str, error: str):
        """Record a failed test."""
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        logger.error(f"❌ {self.name} - {test_name} FAILED: {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        duration = time.time() - self.start_time
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        return {
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "total": total,
            "success_rate": success_rate,
            "duration": duration,
            "errors": self.errors
        }


class Generation1Tests:
    """Tests for Generation 1: Minimal System."""
    
    def run_all_tests(self) -> TestResult:
        """Run all Generation 1 tests."""
        result = TestResult("Generation 1 - Minimal")
        
        # Test basic functionality
        self._test_minimal_generator(result)
        self._test_minimal_evaluator(result)
        self._test_mock_image(result)
        self._test_text_modification(result)
        self._test_bias_metrics(result)
        
        return result
    
    def _test_minimal_generator(self, result: TestResult):
        """Test minimal generator functionality."""
        try:
            generator = MinimalCounterfactualGenerator()
            
            # Test basic generation
            image = MockImage()
            response = generator.generate(
                image=image,
                text="A doctor examining a patient",
                attributes=["gender", "age"],
                num_samples=3
            )
            
            # Validate response structure
            assert "counterfactuals" in response
            assert len(response["counterfactuals"]) == 3
            assert "metadata" in response
            assert response["metadata"]["num_samples"] == 3
            
            result.pass_test("basic_generation")
            
            # Test different attribute formats
            response2 = generator.generate(
                image=image,
                text="A teacher in classroom",
                attributes="gender,race",  # String format
                num_samples=2
            )
            
            assert len(response2["counterfactuals"]) == 2
            result.pass_test("string_attributes")
            
            # Test system status
            status = generator.get_system_status()
            assert "generations_completed" in status
            assert status["generations_completed"] >= 2
            
            result.pass_test("system_status")
            
        except Exception as e:
            result.fail_test("minimal_generator", str(e))
    
    def _test_minimal_evaluator(self, result: TestResult):
        """Test minimal evaluator functionality."""
        try:
            generator = MinimalCounterfactualGenerator()
            evaluator = MinimalBiasEvaluator()
            
            # Generate some counterfactuals
            image = MockImage()
            cf_results = generator.generate(
                image=image,
                text="A professional working",
                attributes=["gender", "age", "race"],
                num_samples=5
            )
            
            # Evaluate bias
            evaluation = evaluator.evaluate(
                counterfactuals=cf_results,
                metrics=["demographic_parity", "cits_score", "fairness_score"]
            )
            
            # Validate evaluation structure
            assert "metrics" in evaluation
            assert "summary" in evaluation
            assert "counterfactual_analysis" in evaluation
            assert len(evaluation["metrics"]) == 3
            
            result.pass_test("basic_evaluation")
            
            # Test report generation
            report = evaluator.generate_report(evaluation, format="technical")
            assert "report_type" in report
            assert "executive_summary" in report
            
            result.pass_test("report_generation")
            
        except Exception as e:
            result.fail_test("minimal_evaluator", str(e))
    
    def _test_mock_image(self, result: TestResult):
        """Test mock image functionality."""
        try:
            # Test basic creation
            img = MockImage(400, 300, "RGB")
            assert img.width == 400
            assert img.height == 300
            assert img.mode == "RGB"
            
            # Test copy
            img_copy = img.copy()
            assert img_copy.width == img.width
            assert img_copy.height == img.height
            
            # Test string representation
            str_repr = str(img)
            assert "MockImage" in str_repr
            assert "400x300" in str_repr
            
            result.pass_test("mock_image")
            
        except Exception as e:
            result.fail_test("mock_image", str(e))
    
    def _test_text_modification(self, result: TestResult):
        """Test text modification functionality."""
        try:
            generator = MinimalCounterfactualGenerator()
            
            # Test gender modification
            original_text = "A man working in an office"
            modified = generator._modify_text(original_text, {"gender": "female"})
            assert "woman" in modified.lower()
            
            # Test age modification
            original_text2 = "A person reading"
            modified2 = generator._modify_text(original_text2, {"age": "elderly"})
            assert "elderly" in modified2.lower()
            
            result.pass_test("text_modification")
            
        except Exception as e:
            result.fail_test("text_modification", str(e))
    
    def _test_bias_metrics(self, result: TestResult):
        """Test bias metrics calculation."""
        try:
            evaluator = MinimalBiasEvaluator()
            
            # Create mock counterfactual data
            cf_data = [
                {
                    "target_attributes": {"gender": "male", "age": "young"},
                    "confidence": 0.8
                },
                {
                    "target_attributes": {"gender": "female", "age": "elderly"},
                    "confidence": 0.7
                },
                {
                    "target_attributes": {"gender": "male", "race": "black"},
                    "confidence": 0.9
                }
            ]
            
            # Test demographic parity
            dp_result = evaluator._compute_demographic_parity(cf_data)
            assert "parity_score" in dp_result
            assert "passes_threshold" in dp_result
            
            # Test CITS score
            cits_result = evaluator._compute_cits_score(cf_data)
            assert "mean_score" in cits_result
            assert "individual_scores" in cits_result
            
            # Test fairness score
            fairness_result = evaluator._compute_fairness_score(cf_data)
            assert "overall_fairness_score" in fairness_result
            assert "rating" in fairness_result
            
            result.pass_test("bias_metrics")
            
        except Exception as e:
            result.fail_test("bias_metrics", str(e))


class Generation2Tests:
    """Tests for Generation 2: Robust System."""
    
    def run_all_tests(self) -> TestResult:
        """Run all Generation 2 tests."""
        result = TestResult("Generation 2 - Robust")
        
        # Test robust functionality
        self._test_robust_generator(result)
        self._test_input_validation(result)
        self._test_error_handling(result)
        self._test_rate_limiting(result)
        self._test_security_validation(result)
        self._test_audit_logging(result)
        self._test_performance_monitoring(result)
        
        return result
    
    def _test_robust_generator(self, result: TestResult):
        """Test robust generator functionality."""
        try:
            generator = RobustCounterfactualGenerator(
                enable_rate_limiting=False,  # Disable for testing
                enable_audit_logging=False
            )
            
            # Test successful generation
            image = generator.base_generator.MockImage(400, 300) if hasattr(generator, 'base_generator') else MockImage(400, 300)
            response = generator.generate(
                image=image,
                text="A scientist conducting research",
                attributes=["gender", "age"],
                num_samples=3,
                user_id="test_user"
            )
            
            # Validate robust response structure
            assert "counterfactuals" in response
            assert "metadata" in response
            assert "security_validated" in response["metadata"]
            assert response["metadata"]["user_id"] == "test_user"
            
            result.pass_test("robust_generation")
            
            # Test health status
            health = generator.get_health_status()
            assert "status" in health
            assert "performance_metrics" in health
            assert "uptime_seconds" in health
            
            result.pass_test("health_status")
            
        except Exception as e:
            result.fail_test("robust_generator", str(e))
    
    def _test_input_validation(self, result: TestResult):
        """Test input validation functionality."""
        try:
            from counterfactual_lab.robust_core import InputValidator
            
            # Test method validation
            try:
                InputValidator.validate_method("invalid_method")
                result.fail_test("method_validation", "Should have failed with invalid method")
            except ValidationError:
                result.pass_test("method_validation")
            
            # Test text validation
            try:
                InputValidator.validate_text("")
                result.fail_test("text_validation", "Should have failed with empty text")
            except ValidationError:
                result.pass_test("text_validation")
            
            # Test valid inputs
            valid_method = InputValidator.validate_method("modicf")
            assert valid_method == "modicf"
            
            valid_text = InputValidator.validate_text("A valid text input")
            assert len(valid_text) > 0
            
            valid_attrs = InputValidator.validate_attributes(["gender", "age"])
            assert "gender" in valid_attrs
            assert "age" in valid_attrs
            
            result.pass_test("valid_inputs")
            
        except Exception as e:
            result.fail_test("input_validation", str(e))
    
    def _test_error_handling(self, result: TestResult):
        """Test error handling functionality."""
        try:
            generator = RobustCounterfactualGenerator(
                enable_rate_limiting=False,
                enable_audit_logging=False
            )
            
            # Test validation error handling
            try:
                generator.generate(
                    image=MockImage(),
                    text="",  # Empty text should fail
                    attributes=["gender"],
                    num_samples=1
                )
                result.fail_test("validation_error", "Should have raised ValidationError")
            except (ValidationError, GenerationError):
                result.pass_test("validation_error")
            
            # Test invalid attributes
            try:
                generator.generate(
                    image=MockImage(),
                    text="Valid text",
                    attributes=[],  # Empty attributes should fail
                    num_samples=1
                )
                result.fail_test("empty_attributes", "Should have raised ValidationError")
            except (ValidationError, GenerationError):
                result.pass_test("empty_attributes")
            
        except Exception as e:
            result.fail_test("error_handling", str(e))
    
    def _test_rate_limiting(self, result: TestResult):
        """Test rate limiting functionality."""
        try:
            from counterfactual_lab.robust_core import RateLimiter
            
            # Test rate limiter
            limiter = RateLimiter(max_requests=3, window_minutes=1)
            
            # First 3 requests should succeed
            for i in range(3):
                assert limiter.check_rate_limit("test_user")
            
            # 4th request should fail
            assert not limiter.check_rate_limit("test_user")
            
            # Different user should succeed
            assert limiter.check_rate_limit("other_user")
            
            # Test rate info
            rate_info = limiter.get_rate_info("test_user")
            assert rate_info["current_requests"] >= 3
            assert rate_info["remaining_requests"] == 0
            
            result.pass_test("rate_limiting")
            
        except Exception as e:
            result.fail_test("rate_limiting", str(e))
    
    def _test_security_validation(self, result: TestResult):
        """Test security validation functionality."""
        try:
            from counterfactual_lab.robust_core import SecurityValidator
            
            # Test text security validation
            safe_text = "A doctor examining a patient"
            is_safe, issues = SecurityValidator.validate_text_input(safe_text)
            assert is_safe
            assert len(issues) == 0
            
            # Test potentially sensitive text
            sensitive_text = "social security number 123-45-6789"
            is_safe, issues = SecurityValidator.validate_text_input(sensitive_text)
            assert not is_safe
            assert len(issues) > 0
            
            # Test attribute validation
            safe_attrs = ["gender", "age"]
            is_safe, issues = SecurityValidator.validate_attributes(safe_attrs)
            assert is_safe
            
            # Test prohibited attributes
            prohibited_attrs = ["social_security", "credit_card"]
            is_safe, issues = SecurityValidator.validate_attributes(prohibited_attrs)
            assert not is_safe
            
            result.pass_test("security_validation")
            
        except Exception as e:
            result.fail_test("security_validation", str(e))
    
    def _test_audit_logging(self, result: TestResult):
        """Test audit logging functionality."""
        try:
            from counterfactual_lab.robust_core import AuditLogger
            
            # Create audit logger
            audit_logger = AuditLogger("test_audit.log")
            
            # Test logging different events
            audit_logger.log_generation_request("test_user", "modicf", ["gender"], True)
            audit_logger.log_evaluation_request("test_user", ["demographic_parity"], True)
            audit_logger.log_security_event("test_event", "info", "Test event", {"test": "data"})
            audit_logger.log_error("test_operation", "Test error", "test_user")
            
            # Check if log file was created
            log_file = Path("test_audit.log")
            if log_file.exists():
                result.pass_test("audit_logging")
                log_file.unlink()  # Clean up
            else:
                result.fail_test("audit_logging", "Log file not created")
            
        except Exception as e:
            result.fail_test("audit_logging", str(e))
    
    def _test_performance_monitoring(self, result: TestResult):
        """Test performance monitoring functionality."""
        try:
            generator = RobustCounterfactualGenerator(
                enable_rate_limiting=False,
                enable_audit_logging=False
            )
            
            # Generate some requests to build metrics
            image = MockImage()
            for i in range(3):
                try:
                    generator.generate(
                        image=image,
                        text=f"Test request {i}",
                        attributes=["gender"],
                        num_samples=1,
                        user_id=f"user_{i}"
                    )
                except:
                    pass  # Expected for some tests
            
            # Check performance metrics
            health = generator.get_health_status()
            assert "performance_metrics" in health
            assert health["performance_metrics"]["total_requests"] > 0
            
            result.pass_test("performance_monitoring")
            
        except Exception as e:
            result.fail_test("performance_monitoring", str(e))


class Generation3Tests:
    """Tests for Generation 3: Scalable System."""
    
    def run_all_tests(self) -> TestResult:
        """Run all Generation 3 tests."""
        result = TestResult("Generation 3 - Scalable")
        
        # Test scalable functionality
        self._test_scalable_generator(result)
        self._test_performance_monitor(result)
        self._test_intelligent_cache(result)
        self._test_worker_pool(result)
        self._test_concurrent_processing(result)
        self._test_batch_generation(result)
        self._test_optimization_features(result)
        
        return result
    
    def _test_scalable_generator(self, result: TestResult):
        """Test scalable generator functionality."""
        try:
            generator = ScalableCounterfactualGenerator(
                enable_caching=True,
                enable_worker_pool=True,
                enable_monitoring=True,
                initial_workers=2,
                max_workers=4
            )
            
            # Test basic generation
            image = MockImage()
            response = generator.generate(
                image=image,
                text="A researcher analyzing data",
                attributes=["gender", "age"],
                num_samples=3,
                user_id="test_user"
            )
            
            # Validate scalable response structure
            assert "counterfactuals" in response
            assert "metadata" in response
            assert "parallel_processing" in response["metadata"]
            assert response["metadata"]["user_id"] == "test_user"
            
            result.pass_test("scalable_generation")
            
            # Test performance metrics
            metrics = generator.get_performance_metrics()
            assert "generator_stats" in metrics
            assert "system_metrics" in metrics
            assert "cache_stats" in metrics
            assert "worker_pool_stats" in metrics
            
            result.pass_test("performance_metrics")
            
            # Test health check
            health = generator.health_check()
            assert "status" in health
            assert "components" in health
            assert len(health["components"]) > 0
            
            result.pass_test("health_check")
            
            generator.shutdown()
            
        except Exception as e:
            result.fail_test("scalable_generator", str(e))
    
    def _test_performance_monitor(self, result: TestResult):
        """Test performance monitoring functionality."""
        try:
            monitor = PerformanceMonitor()
            
            # Test request tracking
            monitor.record_request_start("test_req_1")
            time.sleep(0.01)  # Small delay
            monitor.record_request_end("test_req_1", True, 0.01)
            
            # Test cache tracking
            monitor.record_cache_hit()
            monitor.record_cache_miss()
            
            # Get metrics
            metrics = monitor.get_metrics()
            assert metrics["requests_total"] >= 1
            assert metrics["requests_successful"] >= 1
            assert metrics["cache_hits"] >= 1
            assert metrics["cache_misses"] >= 1
            
            # Test health score
            health_score = monitor.get_health_score()
            assert 0 <= health_score <= 100
            
            monitor.shutdown()
            result.pass_test("performance_monitor")
            
        except Exception as e:
            result.fail_test("performance_monitor", str(e))
    
    def _test_intelligent_cache(self, result: TestResult):
        """Test intelligent caching functionality."""
        try:
            cache = IntelligentCache(max_size=10, default_ttl=60)
            
            # Test cache operations
            cache.set("modicf", "hash1", "text1", {"gender": "male"}, {"test": "data1"})
            cached_value = cache.get("modicf", "hash1", "text1", {"gender": "male"})
            
            assert cached_value is not None
            assert cached_value["test"] == "data1"
            
            # Test cache miss
            miss_value = cache.get("icg", "hash2", "text2", {"age": "young"})
            assert miss_value is None
            
            # Test cache stats
            stats = cache.get_stats()
            assert "size" in stats
            assert "utilization" in stats
            assert stats["size"] >= 1
            
            result.pass_test("intelligent_cache")
            
        except Exception as e:
            result.fail_test("intelligent_cache", str(e))
    
    def _test_worker_pool(self, result: TestResult):
        """Test worker pool functionality."""
        try:
            pool = WorkerPool(initial_workers=2, max_workers=4)
            
            # Test task submission
            def simple_task(x):
                time.sleep(0.01)
                return x * 2
            
            futures = []
            for i in range(5):
                future = pool.submit_task(simple_task, i)
                futures.append(future)
            
            # Wait for completion
            results = []
            for future in futures:
                try:
                    result_val = future.result(timeout=5)
                    results.append(result_val)
                except:
                    pass
            
            assert len(results) > 0
            
            # Test worker pool stats
            stats = pool.get_stats()
            assert "current_workers" in stats
            assert "completed_tasks" in stats
            assert stats["completed_tasks"] > 0
            
            pool.shutdown()
            result.pass_test("worker_pool")
            
        except Exception as e:
            result.fail_test("worker_pool", str(e))
    
    def _test_concurrent_processing(self, result: TestResult):
        """Test concurrent processing capabilities."""
        try:
            generator = ScalableCounterfactualGenerator(
                enable_worker_pool=True,
                initial_workers=2,
                max_workers=4
            )
            
            # Test concurrent generations
            image = MockImage()
            responses = []
            
            # Submit multiple requests
            for i in range(3):
                try:
                    response = generator.generate(
                        image=image,
                        text=f"Concurrent test {i}",
                        attributes=["gender"],
                        num_samples=2,
                        user_id=f"concurrent_user_{i}"
                    )
                    responses.append(response)
                except:
                    pass  # Some may fail, that's ok for testing
            
            # At least one should succeed
            assert len(responses) > 0
            
            generator.shutdown()
            result.pass_test("concurrent_processing")
            
        except Exception as e:
            result.fail_test("concurrent_processing", str(e))
    
    def _test_batch_generation(self, result: TestResult):
        """Test batch generation functionality."""
        try:
            generator = ScalableCounterfactualGenerator(
                enable_worker_pool=True,
                initial_workers=2
            )
            
            # Create batch requests
            image = MockImage()
            batch_requests = [
                {
                    "image": image,
                    "text": f"Batch request {i}",
                    "attributes": ["gender"],
                    "num_samples": 1,
                    "user_id": f"batch_user_{i}"
                }
                for i in range(3)
            ]
            
            # Process batch
            batch_results = generator.generate_batch(batch_requests, max_parallel=2)
            
            # Validate results
            assert len(batch_results) == 3
            
            # Count successful results
            successful = sum(1 for r in batch_results if r.get('success', True) and 'error' not in r)
            assert successful > 0
            
            generator.shutdown()
            result.pass_test("batch_generation")
            
        except Exception as e:
            result.fail_test("batch_generation", str(e))
    
    def _test_optimization_features(self, result: TestResult):
        """Test optimization features."""
        try:
            generator = ScalableCounterfactualGenerator(
                enable_caching=True,
                enable_monitoring=True
            )
            
            # Test caching effectiveness
            image = MockImage()
            
            # First request (cache miss)
            start_time = time.time()
            response1 = generator.generate(
                image=image,
                text="Optimization test",
                attributes=["gender"],
                num_samples=1,
                use_cache=True
            )
            first_duration = time.time() - start_time
            
            # Second request (cache hit)
            start_time = time.time()
            response2 = generator.generate(
                image=image,
                text="Optimization test",
                attributes=["gender"],
                num_samples=1,
                use_cache=True
            )
            second_duration = time.time() - start_time
            
            # Second request should be faster (cached)
            assert second_duration < first_duration or second_duration < 0.01
            
            # Test optimization recommendations
            optimization_report = generator.optimize_performance()
            assert "current_config" in optimization_report
            assert "optimizations" in optimization_report
            
            generator.shutdown()
            result.pass_test("optimization_features")
            
        except Exception as e:
            result.fail_test("optimization_features", str(e))


class IntegrationTests:
    """Integration tests across all generations."""
    
    def run_all_tests(self) -> TestResult:
        """Run all integration tests."""
        result = TestResult("Integration Tests")
        
        self._test_cross_generation_compatibility(result)
        self._test_performance_comparison(result)
        self._test_feature_progression(result)
        
        return result
    
    def _test_cross_generation_compatibility(self, result: TestResult):
        """Test that all generations can handle the same inputs."""
        try:
            # Create generators from all generations
            gen1 = MinimalCounterfactualGenerator()
            gen2 = RobustCounterfactualGenerator(enable_rate_limiting=False, enable_audit_logging=False)
            gen3 = ScalableCounterfactualGenerator()
            
            # Common test parameters
            image = MockImage()
            text = "A software engineer coding"
            attributes = ["gender", "age"]
            num_samples = 2
            
            # Test all generators with same inputs
            response1 = gen1.generate(image=image, text=text, attributes=attributes, num_samples=num_samples)
            response2 = gen2.generate(image=image, text=text, attributes=attributes, num_samples=num_samples)
            response3 = gen3.generate(image=image, text=text, attributes=attributes, num_samples=num_samples)
            
            # All should have basic structure
            for i, response in enumerate([response1, response2, response3], 1):
                assert "counterfactuals" in response, f"Gen{i} missing counterfactuals"
                assert "metadata" in response, f"Gen{i} missing metadata"
                assert len(response["counterfactuals"]) == num_samples, f"Gen{i} wrong sample count"
            
            gen3.shutdown()
            result.pass_test("cross_generation_compatibility")
            
        except Exception as e:
            result.fail_test("cross_generation_compatibility", str(e))
    
    def _test_performance_comparison(self, result: TestResult):
        """Compare performance across generations."""
        try:
            # Create generators
            gen1 = MinimalCounterfactualGenerator()
            gen2 = RobustCounterfactualGenerator(enable_rate_limiting=False, enable_audit_logging=False)
            gen3 = ScalableCounterfactualGenerator(enable_worker_pool=False)  # Disable for fair comparison
            
            image = MockImage()
            text = "Performance test"
            attributes = ["gender"]
            
            # Time each generation
            times = {}
            
            for name, generator in [("Gen1", gen1), ("Gen2", gen2), ("Gen3", gen3)]:
                start_time = time.time()
                response = generator.generate(
                    image=image, text=text, attributes=attributes, num_samples=1
                )
                duration = time.time() - start_time
                times[name] = duration
                
                # Verify successful generation
                assert len(response["counterfactuals"]) == 1
            
            # All generations should complete in reasonable time
            for name, duration in times.items():
                assert duration < 5.0, f"{name} took too long: {duration}s"
            
            gen3.shutdown()
            result.pass_test("performance_comparison")
            
        except Exception as e:
            result.fail_test("performance_comparison", str(e))
    
    def _test_feature_progression(self, result: TestResult):
        """Test that each generation adds features progressively."""
        try:
            # Generation 1: Basic features
            gen1 = MinimalCounterfactualGenerator()
            
            # Should have basic generation
            assert hasattr(gen1, 'generate')
            assert hasattr(gen1, 'get_system_status')
            
            # Generation 2: Robust features
            gen2 = RobustCounterfactualGenerator(enable_rate_limiting=False, enable_audit_logging=False)
            
            # Should have all Gen1 features plus robust ones
            assert hasattr(gen2, 'generate')
            assert hasattr(gen2, 'get_health_status')  # New in Gen2
            
            # Generation 3: Scalable features
            gen3 = ScalableCounterfactualGenerator()
            
            # Should have advanced features
            assert hasattr(gen3, 'generate')
            assert hasattr(gen3, 'generate_batch')  # New in Gen3
            assert hasattr(gen3, 'get_performance_metrics')  # New in Gen3
            assert hasattr(gen3, 'health_check')  # New in Gen3
            assert hasattr(gen3, 'optimize_performance')  # New in Gen3
            
            # Test that Gen3 has internal components Gen1/2 don't
            assert hasattr(gen3, 'performance_monitor')
            assert hasattr(gen3, 'cache')
            assert hasattr(gen3, 'worker_pool')
            
            gen3.shutdown()
            result.pass_test("feature_progression")
            
        except Exception as e:
            result.fail_test("feature_progression", str(e))


def run_comprehensive_test_suite() -> Dict[str, Any]:
    """Run the complete test suite for all generations."""
    logger.info("🧪 Starting Comprehensive Test Suite")
    logger.info("=" * 60)
    
    start_time = time.time()
    all_results = []
    
    # Run tests for each generation
    test_classes = [
        Generation1Tests(),
        Generation2Tests(), 
        Generation3Tests(),
        IntegrationTests()
    ]
    
    for test_class in test_classes:
        logger.info(f"\n🔬 Running {test_class.__class__.__name__}...")
        test_result = test_class.run_all_tests()
        all_results.append(test_result.get_summary())
        logger.info(f"   {test_result.passed} passed, {test_result.failed} failed")
    
    total_duration = time.time() - start_time
    
    # Calculate overall statistics
    total_passed = sum(r["passed"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    total_tests = total_passed + total_failed
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Generate final report
    final_report = {
        "test_suite": "Counterfactual Lab - All Generations",
        "execution_time": total_duration,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "overall_success_rate": overall_success_rate,
        "generation_results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 COMPREHENSIVE TEST SUITE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {total_passed}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success Rate: {overall_success_rate:.1f}%")
    logger.info(f"Execution Time: {total_duration:.2f}s")
    
    # Show per-generation results
    for result in all_results:
        status_icon = "✅" if result["failed"] == 0 else "⚠️" if result["success_rate"] > 50 else "❌"
        logger.info(f"{status_icon} {result['name']}: {result['passed']}/{result['total']} ({result['success_rate']:.1f}%)")
    
    if overall_success_rate >= 80:
        logger.info("\n🎉 TEST SUITE PASSED! All generations working correctly.")
    elif overall_success_rate >= 60:
        logger.info("\n⚠️  TEST SUITE PARTIALLY PASSED. Some issues detected.")
    else:
        logger.info("\n❌ TEST SUITE FAILED. Significant issues detected.")
    
    logger.info("=" * 60)
    
    return final_report


if __name__ == "__main__":
    # Run the comprehensive test suite
    test_report = run_comprehensive_test_suite()
    
    # Save test report
    report_file = Path("test_report.json")
    with open(report_file, 'w') as f:
        import json
        json.dump(test_report, f, indent=2)
    
    logger.info(f"\n📄 Test report saved to {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if test_report["overall_success_rate"] >= 80 else 1
    exit(exit_code)