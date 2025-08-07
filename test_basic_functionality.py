#!/usr/bin/env python3
"""Basic functionality test for Multimodal Counterfactual Lab."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from PIL import Image
import tempfile
import traceback
from pathlib import Path

from counterfactual_lab import CounterfactualGenerator, BiasEvaluator


def create_test_image():
    """Create a simple test image."""
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, 'RGB')
    return image


def test_generator_initialization():
    """Test generator initialization."""
    print("🧪 Testing generator initialization...")
    
    try:
        # Test with CPU device
        gen = CounterfactualGenerator(
            method="modicf", 
            device="cpu", 
            enable_optimization=False
        )
        print(f"  ✅ CPU generator initialized: {gen.method}")
        
        # Test with ICG method
        gen_icg = CounterfactualGenerator(
            method="icg", 
            device="cpu", 
            enable_optimization=False
        )
        print(f"  ✅ ICG generator initialized: {gen_icg.method}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Generator initialization failed: {e}")
        return False


def test_basic_generation():
    """Test basic counterfactual generation."""
    print("\n🧪 Testing basic generation...")
    
    try:
        gen = CounterfactualGenerator(
            method="modicf", 
            device="cpu", 
            enable_optimization=False
        )
        
        # Create test inputs
        image = create_test_image()
        text = "A person standing in a room"
        attributes = ["gender", "age"]
        
        # Generate counterfactuals
        result = gen.generate(
            image=image,
            text=text,
            attributes=attributes,
            num_samples=2
        )
        
        # Validate result structure
        assert "method" in result
        assert "original_image" in result
        assert "original_text" in result
        assert "counterfactuals" in result
        assert "metadata" in result
        
        assert result["method"] == "modicf"
        assert result["original_text"] == text
        assert len(result["counterfactuals"]) == 2
        
        print(f"  ✅ Generated {len(result['counterfactuals'])} counterfactuals")
        print(f"  ✅ Generation time: {result['metadata']['generation_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic generation failed: {e}")
        traceback.print_exc()
        return False


def test_validation():
    """Test input validation."""
    print("\n🧪 Testing input validation...")
    
    try:
        gen = CounterfactualGenerator(device="cpu", enable_optimization=False)
        
        # Test with valid inputs
        image = create_test_image()
        result = gen.generate(
            image=image,
            text="Valid text",
            attributes="gender,age",  # Test string format
            num_samples=1
        )
        print("  ✅ Valid inputs processed successfully")
        
        # Test with invalid num_samples
        try:
            gen.generate(
                image=image,
                text="Valid text",
                attributes=["gender"],
                num_samples=0  # Invalid
            )
            print("  ❌ Should have failed with invalid num_samples")
            return False
        except Exception:
            print("  ✅ Invalid num_samples correctly rejected")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        return False


def test_bias_evaluator():
    """Test bias evaluator."""
    print("\n🧪 Testing bias evaluator...")
    
    try:
        # Create mock model
        class MockModel:
            def __init__(self):
                self.name = "test-model"
        
        model = MockModel()
        evaluator = BiasEvaluator(model)
        
        # Create mock counterfactual data
        counterfactual_data = {
            "method": "modicf",
            "original_text": "A person working",
            "counterfactuals": [
                {
                    "target_attributes": {"gender": "male"},
                    "generated_text": "A man working",
                    "confidence": 0.8
                },
                {
                    "target_attributes": {"gender": "female"},
                    "generated_text": "A woman working",
                    "confidence": 0.9
                }
            ]
        }
        
        # Evaluate bias
        results = evaluator.evaluate(
            counterfactual_data,
            ["demographic_parity", "cits_score"]
        )
        
        # Validate results
        assert "metrics" in results
        assert "summary" in results
        assert "demographic_parity" in results["metrics"]
        assert "cits_score" in results["metrics"]
        
        print(f"  ✅ Bias evaluation completed")
        print(f"  ✅ Fairness score: {results['summary']['overall_fairness_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Bias evaluator test failed: {e}")
        traceback.print_exc()
        return False


def test_storage_and_caching():
    """Test storage and caching functionality."""
    print("\n🧪 Testing storage and caching...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = CounterfactualGenerator(
                device="cpu",
                enable_optimization=False,
                storage_dir=temp_dir,
                cache_dir=os.path.join(temp_dir, "cache")
            )
            
            # Generate with saving enabled
            image = create_test_image()
            result = gen.generate(
                image=image,
                text="Test storage",
                attributes=["gender"],
                num_samples=1,
                save_results=True
            )
            
            # Check if experiment ID was created
            if "saved_experiment_id" in result["metadata"]:
                print(f"  ✅ Results saved with ID: {result['metadata']['saved_experiment_id']}")
            
            # Test cache statistics
            cache_stats = gen.cache_manager.get_cache_stats() if gen.cache_manager else {}
            print(f"  ✅ Cache stats available: {bool(cache_stats)}")
            
            # Test storage statistics
            storage_stats = gen.storage_manager.get_storage_stats()
            print(f"  ✅ Storage stats available: {bool(storage_stats)}")
            
            return True
        
    except Exception as e:
        print(f"  ❌ Storage/caching test failed: {e}")
        return False


def test_system_status():
    """Test system status reporting."""
    print("\n🧪 Testing system status...")
    
    try:
        gen = CounterfactualGenerator(device="cpu", enable_optimization=False)
        
        status = gen.get_system_status()
        
        # Validate status structure
        assert "generator" in status
        assert "timestamp" in status
        
        print("  ✅ System status generated successfully")
        print(f"  ✅ Generator method: {status['generator']['method']}")
        print(f"  ✅ Device: {status['generator']['device']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ System status test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Multimodal Counterfactual Lab Basic Tests\n")
    
    tests = [
        test_generator_initialization,
        test_basic_generation,
        test_validation,
        test_bias_evaluator,
        test_storage_and_caching,
        test_system_status
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  💥 Test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())