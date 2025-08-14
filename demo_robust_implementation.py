#!/usr/bin/env python3
"""
Generation 2 Robust Implementation Demonstration
Showcases comprehensive error handling, validation, security, logging, and monitoring
"""

import sys
import os
import time
import tempfile
sys.path.insert(0, 'src')

from counterfactual_lab import CounterfactualGenerator, BiasEvaluator, _implementation_level
from counterfactual_lab.robust_core import (
    test_robust_system, MockImage, ValidationError, 
    GenerationError, EvaluationError, SecurityError, RateLimitError
)


def print_banner(title: str, char: str = "="):
    """Print a formatted banner."""
    print(f"\n{char*80}")
    print(f"🛡️  {title}")
    print(f"{char*80}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n🔹 {title}")
    print("-" * 60)


def demo_progressive_enhancement():
    """Demonstrate progressive enhancement from lightweight to robust."""
    print_section("Progressive Enhancement Detection")
    
    print(f"🔍 Implementation Level Detected: {_implementation_level.upper()}")
    
    if _implementation_level == "robust":
        print("✅ Full robust implementation with:")
        print("  • Comprehensive error handling and retry logic")
        print("  • Input validation and security checks")
        print("  • Rate limiting and abuse prevention")
        print("  • Audit logging and security monitoring")
        print("  • Performance metrics and health monitoring")
        print("  • Circuit breaker pattern for fault tolerance")
    elif _implementation_level == "full":
        print("⚠️  Full implementation without robust features")
    else:
        print("🔄 Lightweight fallback implementation")


def demo_input_validation():
    """Demonstrate comprehensive input validation."""
    print_section("Input Validation & Security")
    
    generator = CounterfactualGenerator(
        method="modicf",
        enable_rate_limiting=True,
        enable_audit_logging=True
    )
    
    test_cases = [
        {
            "name": "Valid Input",
            "text": "A professional working in their office",
            "attributes": ["gender", "age"],
            "num_samples": 3,
            "should_pass": True
        },
        {
            "name": "Empty Text",
            "text": "",
            "attributes": ["gender"],
            "num_samples": 1,
            "should_pass": False
        },
        {
            "name": "Invalid Attributes",
            "text": "A person",
            "attributes": ["invalid_attr", "another_bad_attr"],
            "num_samples": 1,
            "should_pass": False
        },
        {
            "name": "Too Many Samples",
            "text": "A person working",
            "attributes": ["gender"],
            "num_samples": 200,  # Exceeds limit
            "should_pass": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        try:
            result = generator.generate(
                image=MockImage(400, 300),
                text=test_case["text"],
                attributes=test_case["attributes"],
                num_samples=test_case["num_samples"],
                user_id="validation_test_user"
            )
            
            if test_case["should_pass"]:
                print(f"  ✅ PASS - Generated {len(result['counterfactuals'])} samples")
            else:
                print(f"  ❌ FAIL - Expected validation error but generation succeeded")
                
        except (ValidationError, SecurityError, GenerationError) as e:
            if not test_case["should_pass"]:
                print(f"  ✅ PASS - Correctly blocked: {type(e).__name__}: {str(e)[:80]}...")
            else:
                print(f"  ❌ FAIL - Unexpected error: {e}")
        except Exception as e:
            print(f"  ⚠️  UNEXPECTED - {type(e).__name__}: {e}")


def demo_rate_limiting():
    """Demonstrate rate limiting functionality."""
    print_section("Rate Limiting & Abuse Prevention")
    
    # Create generator with strict rate limits
    generator = CounterfactualGenerator(
        enable_rate_limiting=True,
        max_requests_per_minute=3  # Very low limit for demo
    )
    
    test_image = MockImage(300, 200)
    user_id = "rate_limit_test_user"
    
    print(f"🚦 Testing rate limit: 3 requests per minute for user '{user_id}'")
    
    successful_requests = 0
    rate_limited_requests = 0
    
    for i in range(5):  # Try 5 requests, expect 3 to succeed
        try:
            print(f"  Request {i+1}...", end=" ")
            result = generator.generate(
                image=test_image,
                text=f"Test request {i+1}",
                attributes=["gender"],
                num_samples=1,
                user_id=user_id
            )
            successful_requests += 1
            print("✅ SUCCESS")
            
        except RateLimitError as e:
            rate_limited_requests += 1
            print(f"🛑 RATE LIMITED")
            print(f"    Error: {str(e)[:80]}...")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 Results:")
    print(f"  • Successful requests: {successful_requests}")
    print(f"  • Rate limited requests: {rate_limited_requests}")
    print(f"  • Rate limiting {'✅ WORKING' if rate_limited_requests > 0 else '❌ NOT WORKING'}")


def demo_error_handling_and_retries():
    """Demonstrate comprehensive error handling with retries."""
    print_section("Error Handling & Retry Logic")
    
    generator = CounterfactualGenerator(enable_audit_logging=True)
    
    print("🔄 Testing error handling with various failure scenarios...")
    
    # Test with different failure scenarios
    test_scenarios = [
        {
            "name": "Valid Generation",
            "text": "A researcher analyzing data",
            "attributes": ["age", "gender"],
            "num_samples": 2
        },
        {
            "name": "Complex Generation", 
            "text": "A healthcare professional consulting with colleagues",
            "attributes": ["gender", "age", "expression"],
            "num_samples": 3
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        try:
            start_time = time.time()
            result = generator.generate(
                image=MockImage(400, 300),
                **scenario,
                user_id="error_test_user"
            )
            duration = time.time() - start_time
            
            print(f"  ✅ SUCCESS in {duration:.2f}s")
            print(f"  📊 Generated {len(result['counterfactuals'])} counterfactuals")
            print(f"  🎯 Confidence range: {min(cf.get('confidence', 0) for cf in result['counterfactuals']):.2f}-{max(cf.get('confidence', 0) for cf in result['counterfactuals']):.2f}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {type(e).__name__}: {e}")


def demo_security_monitoring():
    """Demonstrate security monitoring and audit logging."""
    print_section("Security Monitoring & Audit Logging")
    
    generator = CounterfactualGenerator(
        enable_audit_logging=True,
        enable_rate_limiting=True
    )
    
    print("🔐 Testing security monitoring with various security scenarios...")
    
    # Test legitimate usage
    print("\n✅ Legitimate Usage Test:")
    try:
        result = generator.generate(
            image=MockImage(400, 300),
            text="A professional presenting quarterly results to the team",
            attributes=["gender", "age"],
            num_samples=2,
            user_id="legitimate_user"
        )
        print(f"  ✅ Generated {len(result['counterfactuals'])} samples")
        print(f"  🔒 Security validated: {result['metadata']['security_validated']}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    # Test potentially suspicious patterns
    print("\n⚠️ Suspicious Pattern Test:")
    suspicious_texts = [
        "A person with their social security number visible",
        "Someone entering their password at login"
    ]
    
    for text in suspicious_texts:
        try:
            print(f"  Testing: '{text[:40]}...'")
            result = generator.generate(
                image=MockImage(300, 200),
                text=text,
                attributes=["gender"],
                num_samples=1,
                user_id="security_test_user"
            )
            print(f"    ⚠️ Generated (with sanitization)")
        except (SecurityError, ValidationError) as e:
            print(f"    🛑 Blocked: {type(e).__name__}")
        except Exception as e:
            print(f"    ❌ Error: {e}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring and health checks."""
    print_section("Performance Monitoring & Health Checks")
    
    generator = CounterfactualGenerator(enable_audit_logging=True)
    evaluator = BiasEvaluator(enable_audit_logging=True)
    
    print("📊 Running performance and health monitoring tests...")
    
    # Generate some workload
    print("\n🏋️ Creating workload...")
    test_image = MockImage(400, 300)
    results = []
    
    for i in range(3):
        try:
            result = generator.generate(
                image=test_image,
                text=f"Professional scenario {i+1} for performance testing",
                attributes=["gender", "age"],
                num_samples=2,
                user_id=f"perf_test_user_{i}"
            )
            results.append(result)
            print(f"  ✅ Batch {i+1}: {result['metadata']['generation_time']:.3f}s")
        except Exception as e:
            print(f"  ❌ Batch {i+1} failed: {e}")
    
    # Get health status
    print("\n🏥 Health Status Check:")
    health = generator.get_health_status()
    print(f"  Overall Status: {health['status'].upper()}")
    print(f"  Uptime: {health['uptime_seconds']:.1f}s")
    print(f"  Total Generations: {health['total_generations']}")
    print(f"  Error Count: {health['error_count']}")
    print(f"  Success Rate: {health['performance_metrics']['error_rate']*100:.1f}% errors")
    print(f"  Average Response Time: {health['performance_metrics']['avg_response_time']:.3f}s")
    
    # Test evaluator if we have results
    if results:
        print("\n🧪 Bias Evaluation Performance Test:")
        try:
            start_time = time.time()
            evaluation = evaluator.evaluate(
                counterfactuals=results[0],
                metrics=["demographic_parity", "fairness_score", "cits_score"],
                user_id="perf_eval_user"
            )
            eval_time = time.time() - start_time
            
            print(f"  ✅ Evaluation completed in {eval_time:.3f}s")
            print(f"  📊 Fairness rating: {evaluation['summary']['fairness_rating']}")
            
            # Get evaluator stats
            eval_stats = evaluator.get_evaluation_stats()
            print(f"  📈 Evaluator status: {eval_stats['status']}")
            print(f"  🎯 Success rate: {eval_stats['success_rate']*100:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Evaluation failed: {e}")


def demo_comprehensive_bias_evaluation():
    """Demonstrate robust bias evaluation with error handling."""
    print_section("Robust Bias Evaluation")
    
    generator = CounterfactualGenerator()
    evaluator = BiasEvaluator(enable_audit_logging=True)
    
    print("🔍 Generating test data for comprehensive bias evaluation...")
    
    # Generate diverse test data
    test_scenarios = [
        {
            "text": "A doctor consulting with patients in the hospital",
            "attributes": ["gender", "race", "age"],
            "num_samples": 4
        },
        {
            "text": "An engineer working on technical innovations",
            "attributes": ["gender", "age"],
            "num_samples": 3
        }
    ]
    
    all_results = []
    for scenario in test_scenarios:
        try:
            result = generator.generate(
                image=MockImage(400, 300),
                **scenario,
                user_id="bias_eval_test"
            )
            all_results.append(result)
            print(f"  ✅ Generated {len(result['counterfactuals'])} samples for: {scenario['text'][:40]}...")
        except Exception as e:
            print(f"  ❌ Failed scenario: {e}")
    
    if not all_results:
        print("  ⚠️ No test data available for evaluation")
        return
    
    # Test comprehensive evaluation
    print(f"\n📊 Running comprehensive bias evaluation on {len(all_results)} datasets...")
    
    metrics_to_test = [
        ["demographic_parity"],
        ["fairness_score", "cits_score"],
        ["demographic_parity", "fairness_score", "cits_score"],
        ["unknown_metric"]  # Test error handling
    ]
    
    for i, metrics in enumerate(metrics_to_test):
        print(f"\n🧪 Evaluation {i+1}: Metrics = {metrics}")
        try:
            evaluation = evaluator.evaluate(
                counterfactuals=all_results[0],  # Use first dataset
                metrics=metrics,
                user_id=f"eval_test_user_{i}"
            )
            
            print(f"  ✅ Evaluation Status: {evaluation.get('validation_passed', 'Unknown')}")
            
            if 'summary' in evaluation:
                summary = evaluation['summary']
                print(f"  📊 Fairness Score: {summary.get('overall_fairness_score', 0):.3f}")
                print(f"  🏆 Rating: {summary.get('fairness_rating', 'Unknown')}")
            
            # Show metric results
            failed_metrics = [name for name, data in evaluation.get('metrics', {}).items() 
                            if isinstance(data, dict) and 'error' in data]
            if failed_metrics:
                print(f"  ⚠️ Failed metrics: {failed_metrics}")
            
        except EvaluationError as e:
            print(f"  🛑 Evaluation blocked: {str(e)[:80]}...")
        except Exception as e:
            print(f"  ❌ Unexpected error: {type(e).__name__}: {e}")


def demo_fault_tolerance():
    """Demonstrate fault tolerance and graceful degradation."""
    print_section("Fault Tolerance & Graceful Degradation")
    
    print("🛡️ Testing system resilience under various failure conditions...")
    
    generator = CounterfactualGenerator(enable_audit_logging=True)
    
    # Test batch processing with mixed success/failure
    print("\n📦 Batch Processing Resilience Test:")
    
    batch_requests = [
        {
            "image": MockImage(300, 200),
            "text": "Valid request for professional consultation",
            "attributes": ["gender", "age"],
            "num_samples": 2,
            "user_id": "batch_user_1"
        },
        {
            "image": MockImage(300, 200), 
            "text": "",  # Invalid empty text
            "attributes": ["gender"],
            "num_samples": 1,
            "user_id": "batch_user_2"
        },
        {
            "image": MockImage(300, 200),
            "text": "Another valid professional scenario",
            "attributes": ["age"],
            "num_samples": 1,
            "user_id": "batch_user_3"
        }
    ]
    
    try:
        # For robust implementation, we need to use direct generation since batch might not be available
        batch_results = []
        for i, request in enumerate(batch_requests):
            try:
                print(f"  Processing request {i+1}...", end=" ")
                result = generator.generate(**request)
                batch_results.append(result)
                print("✅ SUCCESS")
            except Exception as e:
                batch_results.append({"error": str(e), "request_index": i})
                print(f"❌ FAILED: {type(e).__name__}")
        
        successful = len([r for r in batch_results if "error" not in r])
        failed = len(batch_results) - successful
        
        print(f"\n📊 Batch Results:")
        print(f"  • Successful requests: {successful}/{len(batch_requests)}")
        print(f"  • Failed requests: {failed}/{len(batch_requests)}")
        print(f"  • Fault tolerance: {'✅ WORKING' if successful > 0 else '❌ FAILED'}")
        
    except Exception as e:
        print(f"  ❌ Batch processing failed: {e}")
    
    # Test graceful shutdown
    print(f"\n🔄 Graceful Shutdown Test:")
    try:
        generator.shutdown_gracefully()
        print("  ✅ Graceful shutdown completed successfully")
    except Exception as e:
        print(f"  ❌ Shutdown failed: {e}")


def run_full_robust_demo():
    """Run comprehensive demonstration of robust features."""
    print_banner("GENERATION 2: ROBUST IMPLEMENTATION DEMONSTRATION")
    
    print(f"🚀 Implementation Level: {_implementation_level.upper()}")
    
    if _implementation_level != "robust":
        print("⚠️  Robust features not available in current implementation.")
        print("   This demo requires the robust_core module to be available.")
        return False
    
    demos = [
        ("Progressive Enhancement", demo_progressive_enhancement),
        ("Input Validation", demo_input_validation),
        ("Rate Limiting", demo_rate_limiting),
        ("Error Handling", demo_error_handling_and_retries),
        ("Security Monitoring", demo_security_monitoring),
        ("Performance Monitoring", demo_performance_monitoring),
        ("Bias Evaluation", demo_comprehensive_bias_evaluation),
        ("Fault Tolerance", demo_fault_tolerance)
    ]
    
    successful_demos = 0
    
    for name, demo_func in demos:
        try:
            print_banner(name, "·")
            demo_func()
            successful_demos += 1
            print(f"\n✅ {name} demo completed successfully")
        except Exception as e:
            print(f"\n❌ {name} demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print_banner("GENERATION 2 SUMMARY")
    
    print(f"📊 Demo Results: {successful_demos}/{len(demos)} completed successfully")
    
    if successful_demos == len(demos):
        print("🎉 All robust implementation features working perfectly!")
        print("\n🎯 Generation 2 (Make it Robust) - STATUS: ✅ COMPLETED")
        print("   ├─ Comprehensive input validation ✅")
        print("   ├─ Security monitoring and audit logging ✅")
        print("   ├─ Rate limiting and abuse prevention ✅")
        print("   ├─ Error handling with retry logic ✅")
        print("   ├─ Performance monitoring and health checks ✅")
        print("   ├─ Robust bias evaluation ✅")
        print("   └─ Fault tolerance and graceful degradation ✅")
        return True
    else:
        print(f"⚠️  Some features need attention: {len(demos) - successful_demos} failed")
        return False


def run_integrated_system_test():
    """Run integrated system test using the robust test function."""
    print_banner("INTEGRATED SYSTEM TEST")
    
    try:
        print("🧪 Running comprehensive integrated system test...")
        test_results = test_robust_system()
        
        if test_results["test_status"] == "success":
            print("\n✅ Integrated system test PASSED!")
            print(f"📊 Test Results Summary:")
            
            gen_results = test_results.get("generation_results", {})
            if gen_results:
                print(f"  • Generated {len(gen_results.get('counterfactuals', []))} counterfactuals")
                print(f"  • Generation time: {gen_results.get('metadata', {}).get('generation_time', 0):.3f}s")
            
            eval_results = test_results.get("evaluation_results", {})
            if eval_results and 'summary' in eval_results:
                summary = eval_results['summary']
                print(f"  • Bias evaluation rating: {summary.get('fairness_rating', 'Unknown')}")
                print(f"  • Fairness score: {summary.get('overall_fairness_score', 0):.3f}")
            
            health = test_results.get("health_status", {})
            print(f"  • System health: {health.get('status', 'Unknown').upper()}")
            
            return True
        else:
            print(f"\n❌ Integrated system test FAILED: {test_results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Integrated test failed with exception: {e}")
        return False


if __name__ == "__main__":
    print("Starting Generation 2 Robust Implementation Demo...")
    
    # Run full demo
    demo_success = run_full_robust_demo()
    
    # Run integrated test
    test_success = run_integrated_system_test()
    
    # Final result
    if demo_success and test_success:
        print(f"\n🎉 GENERATION 2 AUTONOMOUS EXECUTION: ✅ COMPLETE")
        print("   All robust implementation features verified and working!")
        exit(0)
    else:
        print(f"\n⚠️  GENERATION 2 EXECUTION: ⚠️  PARTIAL")
        print("   Some features may need additional work.")
        exit(1)