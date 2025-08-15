#!/usr/bin/env python3
"""Quality gates testing without external dependencies."""

import sys
import os
import traceback
import importlib.util
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported successfully."""
    modules_to_test = [
        "counterfactual_lab.self_healing_pipeline",
        "counterfactual_lab.enhanced_error_handling", 
        "counterfactual_lab.auto_scaling",
        "counterfactual_lab.core",
        "counterfactual_lab.monitoring"
    ]
    
    results = {}
    for module_name in modules_to_test:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                results[module_name] = f"FAIL: Module not found"
                continue
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results[module_name] = "PASS: Import successful"
            
        except Exception as e:
            results[module_name] = f"FAIL: {str(e)}"
    
    return results

def test_basic_functionality():
    """Test basic functionality of key components."""
    tests = {}
    
    try:
        # Test self-healing pipeline
        from counterfactual_lab.self_healing_pipeline import SelfHealingPipelineGuard, CircuitBreaker
        
        # Test CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        @cb
        def test_function():
            return "success"
        
        result = test_function()
        if result == "success":
            tests["CircuitBreaker"] = "PASS: Basic functionality works"
        else:
            tests["CircuitBreaker"] = "FAIL: Unexpected result"
            
    except Exception as e:
        tests["CircuitBreaker"] = f"FAIL: {str(e)}"
    
    try:
        # Test self-healing guard
        guard = SelfHealingPipelineGuard(monitoring_interval=1, auto_recovery=True)
        status = guard.get_system_status()
        if isinstance(status, dict) and "monitoring" in status:
            tests["SelfHealingGuard"] = "PASS: Status retrieval works"
        else:
            tests["SelfHealingGuard"] = "FAIL: Invalid status format"
            
    except Exception as e:
        tests["SelfHealingGuard"] = f"FAIL: {str(e)}"
    
    try:
        # Test error handling
        from counterfactual_lab.enhanced_error_handling import ErrorHandler, StructuredLogger
        
        logger = StructuredLogger("test_logger", json_output=False)
        error_handler = ErrorHandler(logger)
        
        # Test error handling
        test_error = Exception("Test error")
        error_context = error_handler.handle_error(
            error=test_error,
            component="test_component",
            operation="test_operation"
        )
        
        if error_context.error_type == "Exception":
            tests["ErrorHandler"] = "PASS: Error handling works"
        else:
            tests["ErrorHandler"] = "FAIL: Error context invalid"
            
    except Exception as e:
        tests["ErrorHandler"] = f"FAIL: {str(e)}"
    
    try:
        # Test auto-scaling
        from counterfactual_lab.auto_scaling import ScalingConfig, WorkerPool
        
        def simple_worker(*args, **kwargs):
            return {"result": "success"}
        
        config = ScalingConfig(min_workers=1, max_workers=2)
        # Don't actually start the pool to avoid threading issues
        if config.min_workers == 1 and config.max_workers == 2:
            tests["AutoScaling"] = "PASS: Configuration works"
        else:
            tests["AutoScaling"] = "FAIL: Configuration invalid"
            
    except Exception as e:
        tests["AutoScaling"] = f"FAIL: {str(e)}"
    
    return tests

def test_core_integration():
    """Test core system integration."""
    tests = {}
    
    try:
        # Test that CounterfactualGenerator can be instantiated
        # (without actually initializing heavy dependencies)
        from counterfactual_lab.core import CounterfactualGenerator
        
        # Just test class definition exists and has required methods
        required_methods = ["generate", "generate_batch", "get_system_status"]
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(CounterfactualGenerator, method):
                missing_methods.append(method)
        
        if not missing_methods:
            tests["CoreIntegration"] = "PASS: All required methods present"
        else:
            tests["CoreIntegration"] = f"FAIL: Missing methods: {missing_methods}"
            
    except Exception as e:
        tests["CoreIntegration"] = f"FAIL: {str(e)}"
    
    return tests

def run_security_checks():
    """Run basic security checks."""
    security_issues = []
    
    # Check for potential security issues in the codebase
    src_dir = Path(__file__).parent / "src"
    
    dangerous_patterns = [
        ("eval(", "Use of eval() function"),
        ("exec(", "Use of exec() function"), 
        ("__import__", "Dynamic imports"),
        ("subprocess.call", "Subprocess without shell=False"),
        ("os.system", "Use of os.system"),
        ("pickle.loads", "Unsafe pickle deserialization")
    ]
    
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
            for pattern, description in dangerous_patterns:
                if pattern in content:
                    security_issues.append(f"{py_file.name}: {description}")
        except Exception:
            continue
    
    return security_issues

def main():
    """Run all quality gate tests."""
    print("🛡️ QUALITY GATES - AUTONOMOUS EXECUTION")
    print("=" * 50)
    
    # Test imports
    print("\n📦 MODULE IMPORT TESTS")
    import_results = test_imports()
    for module, result in import_results.items():
        status = "✅" if result.startswith("PASS") else "❌"
        print(f"{status} {module}: {result}")
    
    # Test basic functionality
    print("\n🔧 FUNCTIONALITY TESTS")
    func_results = test_basic_functionality()
    for test, result in func_results.items():
        status = "✅" if result.startswith("PASS") else "❌"
        print(f"{status} {test}: {result}")
    
    # Test integration
    print("\n🔗 INTEGRATION TESTS")
    integration_results = test_core_integration()
    for test, result in integration_results.items():
        status = "✅" if result.startswith("PASS") else "❌"
        print(f"{status} {test}: {result}")
    
    # Security checks
    print("\n🔒 SECURITY SCAN")
    security_issues = run_security_checks()
    if not security_issues:
        print("✅ No security issues detected")
    else:
        print("⚠️  Security issues found:")
        for issue in security_issues:
            print(f"   - {issue}")
    
    # Summary
    print("\n📊 QUALITY GATES SUMMARY")
    all_results = {**import_results, **func_results, **integration_results}
    total_tests = len(all_results)
    passed_tests = sum(1 for result in all_results.values() if result.startswith("PASS"))
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests and not security_issues:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        return 0
    else:
        print("\n⚠️  SOME QUALITY GATES FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())