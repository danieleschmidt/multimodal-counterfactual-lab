"""Run tests for the autonomous SDLC implementations."""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our implementations (dependency-free)
from counterfactual_lab.minimal_core import (
    MinimalCounterfactualGenerator, MinimalBiasEvaluator, MockImage, test_minimal_system
)

from counterfactual_lab.robust_core import (
    RobustCounterfactualGenerator, RobustBiasEvaluator, 
    ValidationError, GenerationError, RateLimitError,
    InputValidator, SecurityValidator, AuditLogger, test_robust_system
)

from counterfactual_lab.scalable_core import (
    ScalableCounterfactualGenerator, PerformanceMonitor, 
    IntelligentCache, WorkerPool, test_scalable_system
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_generation_tests():
    """Run tests for all three generations."""
    logger.info("🚀 AUTONOMOUS SDLC - GENERATION TESTING")
    logger.info("=" * 60)
    
    results = {}
    start_time = time.time()
    
    # Test Generation 1: Minimal System
    logger.info("\n🔬 Testing Generation 1: MAKE IT WORK (Minimal)")
    logger.info("-" * 40)
    try:
        gen1_result = test_minimal_system()
        results["generation_1"] = {
            "status": "PASSED",
            "system": "Minimal - Basic functionality working",
            "features": ["Basic generation", "Simple evaluation", "Mock image handling"],
            "result": gen1_result
        }
        logger.info("✅ Generation 1: PASSED")
    except Exception as e:
        results["generation_1"] = {"status": "FAILED", "error": str(e)}
        logger.error(f"❌ Generation 1: FAILED - {e}")
    
    # Test Generation 2: Robust System
    logger.info("\n🛡️  Testing Generation 2: MAKE IT ROBUST (Reliable)")
    logger.info("-" * 40)
    try:
        gen2_result = test_robust_system()
        results["generation_2"] = {
            "status": "PASSED",
            "system": "Robust - Error handling, security, logging",
            "features": [
                "Input validation", "Error handling", "Rate limiting",
                "Security validation", "Audit logging", "Performance monitoring"
            ],
            "result": gen2_result
        }
        logger.info("✅ Generation 2: PASSED")
    except Exception as e:
        results["generation_2"] = {"status": "FAILED", "error": str(e)}
        logger.error(f"❌ Generation 2: FAILED - {e}")
    
    # Test Generation 3: Scalable System
    logger.info("\n⚡ Testing Generation 3: MAKE IT SCALE (Optimized)")
    logger.info("-" * 40)
    try:
        gen3_result = test_scalable_system()
        results["generation_3"] = {
            "status": "PASSED", 
            "system": "Scalable - Performance, caching, concurrency",
            "features": [
                "Intelligent caching", "Worker pool management", "Performance monitoring",
                "Batch processing", "Concurrent processing", "Auto-scaling", "Health checks"
            ],
            "result": gen3_result
        }
        logger.info("✅ Generation 3: PASSED")
    except Exception as e:
        results["generation_3"] = {"status": "FAILED", "error": str(e)}
        logger.error(f"❌ Generation 3: FAILED - {e}")
    
    # Integration Tests
    logger.info("\n🔗 Testing Integration & Compatibility")
    logger.info("-" * 40)
    try:
        integration_result = test_integration()
        results["integration"] = {
            "status": "PASSED",
            "system": "Cross-generation compatibility",
            "result": integration_result
        }
        logger.info("✅ Integration: PASSED")
    except Exception as e:
        results["integration"] = {"status": "FAILED", "error": str(e)}
        logger.error(f"❌ Integration: FAILED - {e}")
    
    total_time = time.time() - start_time
    
    # Calculate overall results
    passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
    total = len(results)
    success_rate = (passed / total) * 100
    
    # Generate final report
    logger.info("\n" + "=" * 60)
    logger.info("🎯 AUTONOMOUS SDLC EXECUTION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Execution Time: {total_time:.2f}s")
    logger.info(f"Generations Tested: {total}")
    logger.info(f"Successful: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    # Show detailed results
    for name, result in results.items():
        status_icon = "✅" if result["status"] == "PASSED" else "❌"
        system_name = result.get("system", name.replace("_", " ").title())
        logger.info(f"{status_icon} {system_name}")
        
        if "features" in result:
            for feature in result["features"]:
                logger.info(f"   - {feature}")
        
        if result["status"] == "FAILED":
            logger.error(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Final assessment
    if success_rate == 100:
        logger.info("\n🎉 AUTONOMOUS SDLC EXECUTION: COMPLETE SUCCESS!")
        logger.info("All three generations implemented and working correctly.")
        logger.info("✅ Generation 1: MAKE IT WORK - Basic functionality")
        logger.info("✅ Generation 2: MAKE IT ROBUST - Error handling & security") 
        logger.info("✅ Generation 3: MAKE IT SCALE - Performance & optimization")
    elif success_rate >= 75:
        logger.info("\n🎊 AUTONOMOUS SDLC EXECUTION: MOSTLY SUCCESSFUL!")
        logger.info("Most generations working with minor issues.")
    else:
        logger.info("\n⚠️  AUTONOMOUS SDLC EXECUTION: PARTIAL SUCCESS")
        logger.info("Some generations need attention.")
    
    logger.info("=" * 60)
    
    return {
        "execution_summary": {
            "total_time": total_time,
            "success_rate": success_rate,
            "passed": passed,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "generation_results": results
    }


def test_integration():
    """Test integration between all generations."""
    logger.info("Testing cross-generation compatibility...")
    
    # Test same input across all generations
    image = MockImage(400, 300)
    text = "A software engineer developing AI systems"
    attributes = ["gender", "age"]
    num_samples = 2
    
    results = {}
    
    # Test Generation 1
    gen1 = MinimalCounterfactualGenerator()
    result1 = gen1.generate(image=image, text=text, attributes=attributes, num_samples=num_samples)
    results["gen1"] = result1
    
    # Test Generation 2 
    gen2 = RobustCounterfactualGenerator(enable_rate_limiting=False, enable_audit_logging=False)
    result2 = gen2.generate(image=image, text=text, attributes=attributes, num_samples=num_samples)
    results["gen2"] = result2
    
    # Test Generation 3
    gen3 = ScalableCounterfactualGenerator()
    result3 = gen3.generate(image=image, text=text, attributes=attributes, num_samples=num_samples)
    results["gen3"] = result3
    
    # Validate all have same basic structure
    for name, result in results.items():
        assert "counterfactuals" in result, f"{name} missing counterfactuals"
        assert "metadata" in result, f"{name} missing metadata" 
        assert len(result["counterfactuals"]) == num_samples, f"{name} wrong sample count"
    
    gen3.shutdown()
    
    logger.info("✅ All generations handle same input correctly")
    return results


def create_demonstration():
    """Create a demonstration showing the progression through all generations."""
    logger.info("\n🎭 CREATING AUTONOMOUS SDLC DEMONSTRATION")
    logger.info("=" * 60)
    
    demo_text = "A data scientist analyzing machine learning models"
    demo_attributes = ["gender", "race"]
    demo_samples = 3
    demo_image = MockImage(512, 512)
    
    demo_results = {}
    
    # Demonstrate Generation 1
    logger.info("\n📱 Generation 1 Demo: Basic Functionality")
    gen1 = MinimalCounterfactualGenerator()
    start_time = time.time()
    result1 = gen1.generate(image=demo_image, text=demo_text, attributes=demo_attributes, num_samples=demo_samples)
    gen1_time = time.time() - start_time
    
    logger.info(f"   Generated {len(result1['counterfactuals'])} counterfactuals in {gen1_time:.3f}s")
    logger.info(f"   Features: Basic generation, simple evaluation")
    demo_results["generation_1_demo"] = {
        "time": gen1_time,
        "samples": len(result1['counterfactuals']),
        "features": "Basic functionality"
    }
    
    # Demonstrate Generation 2
    logger.info("\n🛡️ Generation 2 Demo: Robust & Secure")
    gen2 = RobustCounterfactualGenerator(enable_audit_logging=True)
    start_time = time.time()
    result2 = gen2.generate(
        image=demo_image, text=demo_text, attributes=demo_attributes, 
        num_samples=demo_samples, user_id="demo_user"
    )
    gen2_time = time.time() - start_time
    
    logger.info(f"   Generated {len(result2['counterfactuals'])} counterfactuals in {gen2_time:.3f}s")
    logger.info(f"   Features: Input validation, error handling, audit logging, security")
    logger.info(f"   Security validated: {result2['metadata']['security_validated']}")
    
    # Test error handling
    try:
        gen2.generate(image=demo_image, text="", attributes=demo_attributes, num_samples=1)
    except (ValidationError, GenerationError):
        logger.info("   ✅ Error handling working (empty text caught)")
    
    demo_results["generation_2_demo"] = {
        "time": gen2_time,
        "samples": len(result2['counterfactuals']),
        "features": "Robust error handling, security, logging"
    }
    
    # Demonstrate Generation 3
    logger.info("\n⚡ Generation 3 Demo: High Performance & Scalable")
    gen3 = ScalableCounterfactualGenerator(enable_caching=True, enable_worker_pool=True)
    
    # First request (cache miss)
    start_time = time.time()
    result3a = gen3.generate(
        image=demo_image, text=demo_text, attributes=demo_attributes,
        num_samples=demo_samples, user_id="demo_user", use_cache=True
    )
    first_time = time.time() - start_time
    
    # Second request (cache hit)
    start_time = time.time()
    result3b = gen3.generate(
        image=demo_image, text=demo_text, attributes=demo_attributes,
        num_samples=demo_samples, user_id="demo_user", use_cache=True
    )
    second_time = time.time() - start_time
    
    # Batch processing demo
    batch_requests = [
        {
            "image": demo_image,
            "text": f"Batch demo request {i}",
            "attributes": ["gender"],
            "num_samples": 2,
            "user_id": f"batch_demo_{i}"
        }
        for i in range(3)
    ]
    
    start_time = time.time()
    batch_results = gen3.generate_batch(batch_requests, max_parallel=2)
    batch_time = time.time() - start_time
    
    successful_batch = sum(1 for r in batch_results if r.get('success', True) and 'error' not in r)
    
    logger.info(f"   Generated {len(result3a['counterfactuals'])} counterfactuals in {first_time:.3f}s (cache miss)")
    logger.info(f"   Generated {len(result3b['counterfactuals'])} counterfactuals in {second_time:.3f}s (cache hit)")
    logger.info(f"   Batch processed {successful_batch}/{len(batch_requests)} requests in {batch_time:.3f}s")
    
    # Show performance metrics
    metrics = gen3.get_performance_metrics()
    if 'system_metrics' in metrics:
        sys_metrics = metrics['system_metrics']
        logger.info(f"   Total requests: {sys_metrics['requests_total']}")
        logger.info(f"   Cache hit rate: {sys_metrics.get('cache_hit_rate', 0):.1f}%")
        logger.info(f"   Avg response time: {sys_metrics['avg_response_time']:.3f}s")
    
    # Health check
    health = gen3.health_check()
    logger.info(f"   System health: {health['status']}")
    
    demo_results["generation_3_demo"] = {
        "first_time": first_time,
        "cached_time": second_time,
        "batch_time": batch_time,
        "batch_success": f"{successful_batch}/{len(batch_requests)}",
        "features": "Caching, concurrency, batch processing, monitoring"
    }
    
    gen3.shutdown()
    
    # Demo summary
    logger.info("\n📊 DEMONSTRATION SUMMARY")
    logger.info("-" * 30)
    logger.info(f"Generation 1: {demo_results['generation_1_demo']['time']:.3f}s - {demo_results['generation_1_demo']['features']}")
    logger.info(f"Generation 2: {demo_results['generation_2_demo']['time']:.3f}s - {demo_results['generation_2_demo']['features']}")  
    logger.info(f"Generation 3: {demo_results['generation_3_demo']['cached_time']:.3f}s - {demo_results['generation_3_demo']['features']}")
    
    return demo_results


if __name__ == "__main__":
    # Run comprehensive testing
    test_results = run_generation_tests()
    
    # Create demonstration
    demo_results = create_demonstration()
    
    # Final summary
    logger.info("\n" + "🎯" * 20)
    logger.info("AUTONOMOUS SDLC EXECUTION COMPLETE")
    logger.info("🎯" * 20)
    
    success_rate = test_results["execution_summary"]["success_rate"]
    if success_rate == 100:
        logger.info("🏆 PERFECT EXECUTION: All generations working flawlessly!")
    elif success_rate >= 75:
        logger.info("🥇 EXCELLENT EXECUTION: Most features working correctly!")
    else:
        logger.info("🥉 PARTIAL EXECUTION: Some components need attention.")
    
    logger.info(f"Final Success Rate: {success_rate:.1f}%")
    
    # Save results
    import json
    with open("autonomous_sdlc_results.json", "w") as f:
        json.dump({
            "test_results": test_results,
            "demo_results": demo_results
        }, f, indent=2, default=str)
    
    logger.info("📄 Results saved to autonomous_sdlc_results.json")