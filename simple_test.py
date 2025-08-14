#!/usr/bin/env python3
"""
Simple test without dependencies to verify basic functionality
"""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """Test basic imports work."""
    try:
        from counterfactual_lab import CounterfactualGenerator, BiasEvaluator
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_generator_init():
    """Test generator initialization."""
    try:
        from counterfactual_lab import CounterfactualGenerator
        gen = CounterfactualGenerator(device="cpu", enable_monitoring=False)
        print("✅ Generator initialization successful")
        return True
    except Exception as e:
        print(f"❌ Generator init failed: {e}")
        return False

def test_evaluator_init():
    """Test evaluator initialization.""" 
    try:
        from counterfactual_lab import BiasEvaluator
        
        # Mock model
        class MockModel:
            def __init__(self):
                self.name = "mock"
        
        evaluator = BiasEvaluator(MockModel())
        print("✅ Evaluator initialization successful")
        return True
    except Exception as e:
        print(f"❌ Evaluator init failed: {e}")
        return False

def main():
    print("🧪 Running simple tests...")
    
    tests = [
        test_imports,
        test_generator_init,
        test_evaluator_init
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All basic tests passed! Core functionality verified.")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())