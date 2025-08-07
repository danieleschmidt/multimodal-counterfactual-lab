#!/usr/bin/env python3
"""Advanced usage examples for research and production scenarios."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import json

from counterfactual_lab import CounterfactualGenerator, BiasEvaluator
from counterfactual_lab.optimization import OptimizationConfig
from counterfactual_lab.validators import SafetyValidator


def regulatory_compliance_example():
    """Demonstrate regulatory compliance features for EU AI Act."""
    print("🏛️  Regulatory Compliance Example (EU AI Act)")
    print("=" * 50)
    
    # Initialize with safety checks enabled
    generator = CounterfactualGenerator(
        device="cpu",
        enable_safety_checks=True
    )
    
    # Create mock VLM for evaluation
    class MockRegulatoryModel:
        def __init__(self):
            self.name = "production-vlm-v2.1"
            self.version = "2.1.0"
            self.training_date = "2024-10-15"
    
    model = MockRegulatoryModel()
    evaluator = BiasEvaluator(model)
    
    # Generate test data
    image = Image.new('RGB', (256, 256), color='lightblue')
    
    cf_result = generator.generate(
        image=image,
        text="A financial advisor meeting with clients",
        attributes=["gender", "race", "age"],
        num_samples=5
    )
    
    # Perform comprehensive bias evaluation
    evaluation = evaluator.evaluate(
        cf_result,
        metrics=["demographic_parity", "equalized_odds", "disparate_impact", "cits_score"]
    )
    
    # Generate regulatory report
    report = evaluator.generate_report(
        evaluation,
        format="regulatory",
        export_path="eu_ai_act_compliance_report.json"
    )
    
    print(f"✅ Regulatory compliance assessment completed")
    print(f"📊 Overall fairness score: {evaluation['summary']['overall_fairness_score']:.3f}")
    print(f"🏆 Compliance rating: {evaluation['summary']['fairness_rating']}")
    print(f"⚖️  EU AI Act status: {report['compliance_status']['eu_ai_act']}")
    print(f"📅 Next audit due: {report['compliance_status']['next_audit_due'][:10]}")
    
    # Display key findings and recommendations
    if report.get('executive_summary', {}).get('key_findings'):
        print("🔍 Key findings:")
        for finding in report['executive_summary']['key_findings'][:3]:
            print(f"  - {finding}")
    
    if report.get('executive_summary', {}).get('recommendations'):
        print("💡 Recommendations:")
        for rec in report['executive_summary']['recommendations'][:2]:
            print(f"  - {rec}")
    
    print(f"📄 Full report saved to: eu_ai_act_compliance_report.json")


def research_experiment_example():
    """Demonstrate research-oriented experiment setup."""
    print("\n🔬 Research Experiment Example")
    print("=" * 40)
    
    # Configure for research reproducibility
    experiment_config = {
        "experiment_name": "counterfactual_fairness_study_2025",
        "random_seed": 42,
        "methods": ["modicf", "icg"],
        "attributes": ["gender", "race", "age"],
        "sample_sizes": [5, 10, 20],
        "test_scenarios": [
            "A doctor treating patients",
            "A teacher instructing students", 
            "An engineer solving problems",
            "A lawyer presenting a case"
        ]
    }
    
    results = {}
    
    for method in experiment_config["methods"]:
        print(f"🧪 Testing method: {method}")
        
        generator = CounterfactualGenerator(
            method=method,
            device="cpu",
            enable_optimization=True
        )
        
        method_results = []
        
        for scenario in experiment_config["test_scenarios"][:2]:  # Limit for demo
            for sample_size in experiment_config["sample_sizes"][:2]:  # Limit for demo
                
                # Generate test image
                test_image = Image.new('RGB', (128, 128), 
                                     color=tuple(np.random.randint(0, 255, 3)))
                
                # Run generation
                import time
                start_time = time.time()
                
                cf_result = generator.generate(
                    image=test_image,
                    text=scenario,
                    attributes=experiment_config["attributes"],
                    num_samples=sample_size
                )
                
                generation_time = time.time() - start_time
                
                # Record metrics
                method_results.append({
                    "scenario": scenario,
                    "sample_size": sample_size,
                    "generation_time": generation_time,
                    "num_generated": len(cf_result["counterfactuals"]),
                    "success_rate": 1.0,  # All succeeded in this demo
                })
        
        results[method] = method_results
        print(f"  ✅ {method}: {len(method_results)} test cases completed")
    
    # Analyze results
    print("\n📊 Experimental Results:")
    for method, data in results.items():
        avg_time = sum(r["generation_time"] for r in data) / len(data)
        print(f"  {method}: avg time = {avg_time:.3f}s")
    
    # Save experimental data
    with open("research_experiment_results.json", "w") as f:
        json.dump({
            "config": experiment_config,
            "results": results,
            "analysis_timestamp": "2025-08-07T04:50:00Z"
        }, f, indent=2)
    
    print("💾 Experiment data saved to: research_experiment_results.json")


def production_pipeline_example():
    """Demonstrate production-ready pipeline setup."""
    print("\n🏭 Production Pipeline Example")
    print("=" * 40)
    
    # Production-optimized configuration
    prod_config = OptimizationConfig(
        max_workers=4,
        batch_size=8,
        use_gpu_batching=False,  # CPU for this demo
        enable_memory_optimization=True,
        enable_parallel_processing=True,
        target_image_size=(512, 512)
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize production-ready generator
        generator = CounterfactualGenerator(
            method="modicf",
            device="cpu", 
            enable_optimization=True,
            optimization_config=prod_config,
            storage_dir=f"{temp_dir}/storage",
            cache_dir=f"{temp_dir}/cache"
        )
        
        print("✅ Production generator initialized")
        
        # Simulate production workload
        production_requests = []
        
        for i in range(10):  # Simulating 10 concurrent requests
            # Different image for each request
            img_color = tuple(np.random.randint(100, 255, 3))
            test_image = Image.new('RGB', (256, 256), color=img_color)
            
            production_requests.append({
                "image": test_image,
                "text": f"Production scenario {i+1}: professional in workplace",
                "attributes": ["gender", "age"],
                "num_samples": 2,
                "save_results": True,
                "experiment_id": f"prod_batch_001_item_{i+1:03d}"
            })
        
        # Process production batch with monitoring
        print("🚀 Processing production batch...")
        import time
        start_time = time.time()
        
        results = generator.generate_batch(production_requests)
        
        total_time = time.time() - start_time
        throughput = len(results) / total_time
        
        print(f"✅ Production batch completed:")
        print(f"  📊 Processed: {len(results)} requests")
        print(f"  ⏱️  Total time: {total_time:.2f}s") 
        print(f"  🚀 Throughput: {throughput:.1f} requests/second")
        
        # Get system status for monitoring
        status = generator.get_system_status()
        
        if 'optimization' in status:
            opt_stats = status['optimization']
            print(f"  💾 Memory usage: {opt_stats['memory']['usage_mb']:.1f} MB")
            print(f"  💨 Memory available: {opt_stats['memory']['available_mb']:.1f} MB")
        
        # List all experiments created
        experiments = generator.storage_manager.list_experiments()
        print(f"  🧪 Experiments stored: {len(experiments)}")
        
        # Resource cleanup
        generator.cleanup_resources()
        print("🧹 Production resources cleaned up")


def safety_and_ethics_example():
    """Demonstrate safety and ethical AI features."""
    print("\n🛡️  Safety & Ethics Example")
    print("=" * 40)
    
    # Test ethical validation
    test_cases = [
        {
            "text": "A teacher helping students learn",
            "attributes": ["age", "gender"],
            "expected_safe": True
        },
        {
            "text": "Person with illegal activities mentioned",
            "attributes": ["race"],
            "expected_safe": False
        },
        {
            "text": "Medical professional treating patients",
            "attributes": ["race", "gender", "age"],  # Many demographic changes
            "expected_safe": False  # Should trigger warning for multiple demographic changes
        }
    ]
    
    print("🔍 Running safety validations:")
    
    for i, test_case in enumerate(test_cases):
        print(f"\n  Test {i+1}: {test_case['text'][:30]}...")
        
        is_safe, warnings = SafetyValidator.validate_ethical_use(
            test_case["text"], 
            test_case["attributes"]
        )
        
        status = "✅ SAFE" if is_safe else "⚠️  WARNING" 
        print(f"    Result: {status}")
        
        if warnings:
            for warning in warnings:
                print(f"    - {warning}")
        
        # Test privacy validation
        is_private_safe, privacy_warnings = SafetyValidator.validate_data_privacy()
        
        if privacy_warnings:
            print("    Privacy concerns:")
            for warning in privacy_warnings:
                print(f"    - {warning}")
    
    # Initialize generator with safety enabled
    generator = CounterfactualGenerator(
        device="cpu",
        enable_safety_checks=True
    )
    
    print("\n🛡️  Safety checks enabled in generator")
    
    # Test with safe input
    safe_image = Image.new('RGB', (128, 128), color='green')
    
    result = generator.generate(
        image=safe_image,
        text="A professional working collaboratively",
        attributes=["gender"],  # Single attribute change
        num_samples=1
    )
    
    safety_status = result["metadata"].get("safety_checks_enabled", False)
    print(f"✅ Safe generation completed (safety checks: {safety_status})")


def advanced_caching_example():
    """Demonstrate advanced caching strategies."""
    print("\n💨 Advanced Caching Example")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = CounterfactualGenerator(
            device="cpu",
            cache_dir=f"{temp_dir}/cache"
        )
        
        # Create test image
        test_image = Image.new('RGB', (128, 128), color='blue')
        
        # First generation - should be cached
        print("🔄 First generation (cache miss)...")
        result1 = generator.generate(
            image=test_image,
            text="A consistent test scenario",
            attributes=["gender"],
            num_samples=1
        )
        time1 = result1["metadata"]["generation_time"]
        
        # Second identical generation - should use cache
        print("⚡ Second identical generation (cache hit expected)...")
        result2 = generator.generate(
            image=test_image, 
            text="A consistent test scenario",
            attributes=["gender"],
            num_samples=1
        )
        time2 = result2["metadata"]["generation_time"]
        
        # Check cache effectiveness
        cache_stats = generator.cache_manager.get_cache_stats()
        
        print(f"✅ Cache performance:")
        print(f"  First generation: {time1:.4f}s")
        print(f"  Second generation: {time2:.4f}s")
        print(f"  Speedup: {time1/time2:.1f}x" if time2 > 0 else "  Speedup: ∞")
        print(f"  Cache entries: {cache_stats['total_entries']}")
        print(f"  Cache size: {cache_stats['total_size_mb']:.2f} MB")
        
        # Display cache hit rates if available
        for cache_type in ["generation", "evaluation"]:
            hit_rate_key = f"{cache_type}_hit_rate"
            if hit_rate_key in cache_stats:
                hit_rate = cache_stats[hit_rate_key] * 100
                print(f"  {cache_type.title()} hit rate: {hit_rate:.1f}%")


def main():
    """Run all advanced examples."""
    print("🚀 Multimodal Counterfactual Lab - Advanced Examples")
    print("=" * 70)
    
    advanced_examples = [
        regulatory_compliance_example,
        research_experiment_example,
        production_pipeline_example,
        safety_and_ethics_example,
        advanced_caching_example
    ]
    
    for example in advanced_examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Advanced example failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()  # Add spacing between examples
    
    print("✅ All advanced examples completed!")
    print("\n🎓 What you learned:")
    print("  - Regulatory compliance and audit reporting")  
    print("  - Research experiment design and reproducibility")
    print("  - Production pipeline optimization and monitoring")
    print("  - Safety and ethical AI validation")
    print("  - Advanced caching for performance")
    print("\n📈 Ready for production deployment!")


if __name__ == "__main__":
    main()