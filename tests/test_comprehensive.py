"""Comprehensive test suite for counterfactual generation system."""

import unittest
import tempfile
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from counterfactual_lab import CounterfactualGenerator, BiasEvaluator
from counterfactual_lab.enhanced_core import EnhancedCounterfactualGenerator, EnhancedBiasEvaluator
from counterfactual_lab.production_core import ProductionCounterfactualGenerator
from counterfactual_lab.security import SecurityValidator
from counterfactual_lab.validators import InputValidator
from counterfactual_lab.data.cache import CacheManager
from counterfactual_lab.data.storage import StorageManager
from counterfactual_lab.exceptions import ValidationError, SecurityError


class TestBasicFunctionality(unittest.TestCase):
    """Test basic counterfactual generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        self.test_text = "A person working in an office"
        self.test_attributes = ['gender', 'age']
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            self.test_image_path = f.name
            self.test_image.save(f.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    def test_basic_generation(self):
        """Test basic counterfactual generation."""
        generator = CounterfactualGenerator(method='modicf', device='cpu')
        
        result = generator.generate(
            image=self.test_image,
            text=self.test_text,
            attributes=self.test_attributes,
            num_samples=2
        )
        
        # Verify result structure
        self.assertIn('counterfactuals', result)
        self.assertIn('method', result)
        self.assertIn('metadata', result)
        
        # Verify counterfactuals
        self.assertEqual(len(result['counterfactuals']), 2)
        
        for cf in result['counterfactuals']:
            self.assertIn('target_attributes', cf)
            self.assertIn('confidence', cf)
            self.assertIn('generated_image', cf)
    
    def test_icg_method(self):
        """Test ICG generation method."""
        generator = CounterfactualGenerator(method='icg', device='cpu')
        
        result = generator.generate(
            image=self.test_image,
            text=self.test_text,
            attributes=['gender'],
            num_samples=1
        )
        
        self.assertEqual(result['method'], 'icg')
        self.assertEqual(len(result['counterfactuals']), 1)
    
    def test_bias_evaluation(self):
        """Test bias evaluation functionality."""
        # Create mock model
        class MockModel:
            def __init__(self):
                self.name = 'test-model'
        
        # Generate counterfactuals first
        generator = CounterfactualGenerator(method='modicf', device='cpu')
        cf_result = generator.generate(
            image=self.test_image,
            text=self.test_text,
            attributes=['gender'],
            num_samples=2
        )
        
        # Evaluate bias
        evaluator = BiasEvaluator(MockModel())
        eval_result = evaluator.evaluate(
            cf_result,
            metrics=['demographic_parity', 'cits_score']
        )
        
        # Verify evaluation structure
        self.assertIn('summary', eval_result)
        self.assertIn('metrics', eval_result)
        self.assertIn('overall_fairness_score', eval_result['summary'])
        self.assertIn('fairness_rating', eval_result['summary'])


class TestSecurityFeatures(unittest.TestCase):
    """Test security and validation features."""
    
    def setUp(self):
        """Set up security test fixtures."""
        self.test_image = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            self.safe_image_path = f.name
            self.test_image.save(f.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.safe_image_path):
            os.unlink(self.safe_image_path)
    
    def test_input_validation(self):
        """Test input validation functionality."""
        # Test valid inputs
        method = InputValidator.validate_method('modicf')
        self.assertEqual(method, 'modicf')
        
        attributes = InputValidator.validate_attributes(['gender', 'age'])
        self.assertIn('gender', attributes)
        self.assertIn('age', attributes)
        
        # Test invalid inputs
        with self.assertRaises(ValidationError):
            InputValidator.validate_method('invalid_method')
        
        with self.assertRaises(ValidationError):
            InputValidator.validate_attributes([])
        
        with self.assertRaises(ValidationError):
            InputValidator.validate_num_samples(-1)
    
    def test_security_validation(self):
        """Test security validation features."""
        # Test text sanitization
        malicious_text = '<script>alert("hack")</script>A doctor'
        clean_text = SecurityValidator.sanitize_text_input(malicious_text)
        
        self.assertNotIn('<script>', clean_text)
        self.assertIn('doctor', clean_text.lower())
        
        # Test file validation
        is_valid, message = SecurityValidator.validate_file_path('safe_file.jpg')
        self.assertTrue(is_valid)
        
        is_valid, message = SecurityValidator.validate_file_path('../../../etc/passwd')
        self.assertFalse(is_valid)
        
        # Test image validation
        is_valid, message, metadata = SecurityValidator.validate_image_file(self.safe_image_path)
        self.assertTrue(is_valid)
        self.assertIn('size', metadata)
        self.assertIn('format', metadata)
    
    def test_enhanced_security_generator(self):
        """Test enhanced generator with security features."""
        generator = EnhancedCounterfactualGenerator(
            method='modicf', 
            device='cpu',
            enable_security=True
        )
        
        # Create session
        session_id = generator.create_session('test_user')
        self.assertIsInstance(session_id, str)
        
        # Test secure generation
        result = generator.secure_generate(
            image=self.test_image,
            text='A person working',
            attributes=['gender'],
            num_samples=1,
            user_id='test_user'
        )
        
        self.assertIn('metadata', result)
        self.assertTrue(result['metadata']['security_enabled'])


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.test_images = [
            Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
            for _ in range(3)
        ]
    
    def test_caching_functionality(self):
        """Test caching system performance."""
        cache_manager = CacheManager(max_size_mb=100)
        
        # Test cache operations
        test_data = {'test': 'data', 'number': 42}
        cache_key = 'test/key'
        
        # Cache data
        cache_manager.cache_result(cache_key, test_data)
        
        # Retrieve cached data
        cached_data = cache_manager.get_cached_result(cache_key)
        self.assertEqual(cached_data, test_data)
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        self.assertIn('hit_count', stats)
        self.assertIn('miss_count', stats)
        self.assertIn('size_mb', stats)
    
    def test_storage_functionality(self):
        """Test storage system performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager(base_dir=temp_dir)
            
            # Test image storage
            test_image = self.test_images[0]
            image_info = storage.save_image(test_image, filename='test.png')
            
            self.assertIn('filename', image_info)
            self.assertIn('path', image_info)
            
            # Test loading
            loaded_image = storage.load_image(image_info['path'])
            self.assertEqual(loaded_image.size, test_image.size)
            
            # Test storage stats
            stats = storage.get_storage_stats()
            self.assertIn('total_size_mb', stats)
    
    def test_batch_processing(self):
        """Test batch processing performance."""
        generator = CounterfactualGenerator(method='modicf', device='cpu')
        
        # Create batch requests
        batch_requests = [
            {
                'image': img,
                'text': f'Person {i}',
                'attributes': ['gender'],
                'num_samples': 1
            }
            for i, img in enumerate(self.test_images)
        ]
        
        # Time batch processing
        start_time = time.time()
        results = []
        
        for request in batch_requests:
            result = generator.generate(**request)
            results.append(result)
        
        batch_time = time.time() - start_time
        
        self.assertEqual(len(results), len(batch_requests))
        self.assertLess(batch_time, 10.0)  # Should complete within 10 seconds


class TestProductionFeatures(unittest.TestCase):
    """Test production-ready features."""
    
    def setUp(self):
        """Set up production test fixtures."""
        self.test_image = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
    
    def test_production_generator(self):
        """Test production generator functionality."""
        generator = ProductionCounterfactualGenerator(
            method='modicf',
            device='cpu',
            enable_security=False,
            enable_async=False,
            enable_distributed=False
        )
        
        # Test generation
        result = generator.generate(
            image=self.test_image,
            text='A professional person',
            attributes=['gender'],
            num_samples=1,
            user_id='test_user'
        )
        
        # Verify production metadata
        self.assertIn('production_metadata', result)
        self.assertIn('request_id', result['production_metadata'])
        self.assertIn('processing_method', result['production_metadata'])
        
        # Test system status
        status = generator.get_system_status()
        self.assertIn('performance_metrics', status)
        self.assertIn('configuration', status)
        
        # Test performance report
        report = generator.get_performance_report()
        self.assertIn('summary', report)
        self.assertIn('recommendations', report)
    
    def test_health_monitoring(self):
        """Test health monitoring functionality."""
        generator = ProductionCounterfactualGenerator(
            method='modicf',
            device='cpu'
        )
        
        # Get health status
        health = generator.base_generator.get_health_status()
        
        self.assertIn('overall_status', health)
        self.assertIn('health_score', health)
        self.assertIn('health_checks', health)
        
        # Should be healthy for basic setup
        self.assertIn(health['overall_status'], ['healthy', 'degraded'])
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        generator = ProductionCounterfactualGenerator(
            method='modicf',
            device='cpu'
        )
        
        # Test with invalid input (should handle gracefully)
        with self.assertRaises((ValidationError, ValueError)):
            generator.generate(
                image="nonexistent_file.jpg",
                text="",
                attributes=[],
                num_samples=0
            )


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
    
    def test_complete_workflow(self):
        """Test complete generation and evaluation workflow."""
        # Initialize components
        generator = CounterfactualGenerator(method='modicf', device='cpu')
        
        class MockModel:
            def __init__(self):
                self.name = 'integration-test-model'
        
        evaluator = BiasEvaluator(MockModel())
        
        # Generate counterfactuals
        cf_result = generator.generate(
            image=self.test_image,
            text='A healthcare worker',
            attributes=['gender', 'age'],
            num_samples=3
        )
        
        # Evaluate bias
        eval_result = evaluator.evaluate(
            cf_result,
            metrics=['demographic_parity', 'equalized_odds']
        )
        
        # Generate report
        report = evaluator.generate_report(eval_result, format='summary')
        
        # Verify complete workflow
        self.assertIn('counterfactuals', cf_result)
        self.assertEqual(len(cf_result['counterfactuals']), 3)
        
        self.assertIn('summary', eval_result)
        self.assertIsInstance(report, dict)
        self.assertIn('Report Summary', report)
    
    def test_multi_method_comparison(self):
        """Test comparison between different generation methods."""
        methods = ['modicf', 'icg']
        results = {}
        
        for method in methods:
            generator = CounterfactualGenerator(method=method, device='cpu')
            
            result = generator.generate(
                image=self.test_image,
                text='A scientist in a laboratory',
                attributes=['gender'],
                num_samples=2
            )
            
            results[method] = result
        
        # Verify both methods work
        for method in methods:
            self.assertIn(method, results)
            self.assertEqual(results[method]['method'], method)
            self.assertEqual(len(results[method]['counterfactuals']), 2)


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    
    print("🧪 Running Comprehensive Test Suite...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBasicFunctionality,
        TestSecurityFeatures,
        TestPerformanceOptimization,
        TestProductionFeatures,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True
    )
    
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('\\n')[-2]}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('\\n')[-2]}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED!")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)