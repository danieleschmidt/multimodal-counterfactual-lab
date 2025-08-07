#!/usr/bin/env python3
"""Basic usage examples for Multimodal Counterfactual Lab."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PIL import Image
import numpy as np
from pathlib import Path

from counterfactual_lab import CounterfactualGenerator, BiasEvaluator


def create_sample_image():
    """Create a sample image for demonstration."""
    # Create a simple image with gradient
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            img_array[i, j] = [int(i), int(j), 128]
    
    return Image.fromarray(img_array, 'RGB')


def basic_generation_example():
    """Demonstrate basic counterfactual generation."""
    print("🎯 Basic Generation Example")
    print("=" * 40)
    
    # Initialize generator
    generator = CounterfactualGenerator(
        method="modicf",
        device="cpu",
        enable_optimization=False  # Disabled for this example
    )
    
    # Create or load image
    image = create_sample_image()
    
    # Generate counterfactuals
    result = generator.generate(
        image=image,
        text="A professional working at their computer",
        attributes=["gender", "age"],
        num_samples=3
    )
    
    print(f"✅ Generated {len(result['counterfactuals'])} counterfactuals")
    print(f"📊 Generation time: {result['metadata']['generation_time']:.3f} seconds")
    print(f"🔧 Method used: {result['method']}")
    print(f"💻 Device: {result['metadata']['device']}")
    
    # Display counterfactual details
    for i, cf in enumerate(result['counterfactuals']):
        attrs = cf['target_attributes']
        print(f"  CF {i+1}: {attrs}")
    
    return result


def batch_generation_example():
    """Demonstrate batch processing."""
    print("\n🚀 Batch Generation Example")
    print("=" * 40)
    
    generator = CounterfactualGenerator(device="cpu")
    
    # Prepare multiple requests
    base_image = create_sample_image()
    requests = [
        {
            "image": base_image,
            "text": "A teacher explaining a concept",
            "attributes": ["age", "gender"],
            "num_samples": 2
        },
        {
            "image": base_image,
            "text": "A doctor reviewing medical charts",
            "attributes": ["race", "age"], 
            "num_samples": 2
        },
        {
            "image": base_image,
            "text": "An engineer designing a product",
            "attributes": ["gender"],
            "num_samples": 1
        }
    ]
    
    # Process batch
    results = generator.generate_batch(requests)
    
    print(f"✅ Processed {len(results)} requests")
    total_cfs = sum(len(r['counterfactuals']) for r in results)
    print(f"📊 Total counterfactuals: {total_cfs}")
    
    for i, result in enumerate(results):
        print(f"  Request {i+1}: {len(result['counterfactuals'])} CFs - {result['original_text'][:30]}...")
    
    return results


def bias_evaluation_example():
    """Demonstrate bias evaluation."""
    print("\n🔍 Bias Evaluation Example")
    print("=" * 40)
    
    # Mock model for evaluation
    class MockVLM:
        def __init__(self):
            self.name = "demo-vision-language-model"
    
    model = MockVLM()
    evaluator = BiasEvaluator(model)
    
    # Generate counterfactuals first
    generator = CounterfactualGenerator(device="cpu")
    image = create_sample_image()
    
    cf_data = generator.generate(
        image=image,
        text="A person presenting to colleagues",
        attributes=["gender", "race"],
        num_samples=4
    )
    
    # Evaluate for bias
    evaluation_results = evaluator.evaluate(
        cf_data,
        metrics=["demographic_parity", "equalized_odds", "cits_score"]
    )
    
    print(f"✅ Bias evaluation completed")
    print(f"📊 Overall fairness score: {evaluation_results['summary']['overall_fairness_score']:.3f}")
    print(f"🏆 Rating: {evaluation_results['summary']['fairness_rating']}")
    print(f"📋 Metrics evaluated: {list(evaluation_results['metrics'].keys())}")
    
    # Display key findings
    for metric, data in evaluation_results['metrics'].items():
        if isinstance(data, dict) and 'passes_threshold' in data:
            status = "✅ PASS" if data['passes_threshold'] else "❌ FAIL"
            print(f"  {metric}: {status}")
    
    return evaluation_results


def storage_and_persistence_example():
    """Demonstrate storage and persistence features."""
    print("\n💾 Storage & Persistence Example")
    print("=" * 40)
    
    # Create temporary directory for this example
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = CounterfactualGenerator(
            device="cpu",
            storage_dir=temp_dir,
            cache_dir=f"{temp_dir}/cache"
        )
        
        image = create_sample_image()
        
        # Generate with saving enabled
        result = generator.generate(
            image=image,
            text="A researcher analyzing data",
            attributes=["gender", "age"],
            num_samples=2,
            save_results=True,
            experiment_id="demo_experiment_001"
        )
        
        if "saved_experiment_id" in result["metadata"]:
            exp_id = result["metadata"]["saved_experiment_id"]
            print(f"✅ Results saved with ID: {exp_id}")
        
        # Show storage statistics
        storage_stats = generator.storage_manager.get_storage_stats()
        print(f"📁 Storage usage: {storage_stats['total_size_mb']:.2f} MB")
        
        # Show cache statistics
        if generator.cache_manager:
            cache_stats = generator.cache_manager.get_cache_stats()
            print(f"🗃️  Cache entries: {cache_stats['total_entries']}")
            print(f"💨 Cache usage: {cache_stats['total_size_mb']:.2f} MB")
        
        # List experiments
        experiments = generator.storage_manager.list_experiments()
        print(f"🧪 Experiments stored: {len(experiments)}")


def performance_optimization_example():
    """Demonstrate performance optimization features."""
    print("\n⚡ Performance Optimization Example")
    print("=" * 40)
    
    # Initialize with optimization enabled
    generator = CounterfactualGenerator(
        device="cpu",
        enable_optimization=True
    )
    
    # Get optimization stats
    if generator.optimizer:
        stats = generator.optimizer.get_optimization_stats()
        print(f"✅ Optimization enabled")
        print(f"👥 Max workers: {stats['config']['max_workers']}")
        print(f"📦 Batch size: {stats['config']['batch_size']}")
        print(f"💾 Memory usage: {stats['memory']['usage_mb']:.1f} MB")
        
        # Update optimization settings
        generator.optimize_performance(batch_size=8, max_workers=4)
        print("⚙️  Updated optimization settings")
    else:
        print("❌ Optimization not available")
    
    # Test performance with batch processing
    import time
    
    image = create_sample_image()
    requests = [
        {
            "image": image,
            "text": f"Person {i} working",
            "attributes": ["gender"],
            "num_samples": 1
        }
        for i in range(5)
    ]
    
    start_time = time.time()
    results = generator.generate_batch(requests)
    batch_time = time.time() - start_time
    
    print(f"⏱️  Batch processing time: {batch_time:.3f}s ({batch_time/len(requests):.3f}s per item)")


def system_monitoring_example():
    """Demonstrate system monitoring and diagnostics."""
    print("\n📊 System Monitoring Example")  
    print("=" * 40)
    
    generator = CounterfactualGenerator(device="cpu")
    
    # Get comprehensive system status
    status = generator.get_system_status()
    
    print("🖥️  System Status:")
    print(f"  Generator method: {status['generator']['method']}")
    print(f"  Device: {status['generator']['device']}")
    print(f"  Cache enabled: {status['generator']['cache_enabled']}")
    print(f"  Optimization enabled: {status['generator']['optimization_enabled']}")
    
    if 'diagnostics' in status and 'error' not in status['diagnostics']:
        diag = status['diagnostics']
        health = diag.get('health', {})
        print(f"  Overall health: {health.get('overall_status', 'unknown')}")
        print(f"  CPU usage: {health.get('cpu_usage', 0):.1f}%")
        print(f"  Memory usage: {health.get('memory_usage', 0):.1f}%")
        print(f"  GPU available: {health.get('gpu_available', False)}")
    
    # Cleanup resources
    generator.cleanup_resources()
    print("🧹 Resources cleaned up")


def main():
    """Run all examples."""
    print("🧪 Multimodal Counterfactual Lab - Usage Examples")
    print("=" * 60)
    
    examples = [
        basic_generation_example,
        batch_generation_example, 
        bias_evaluation_example,
        storage_and_persistence_example,
        performance_optimization_example,
        system_monitoring_example
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All examples completed!")
    print("\n📚 Next steps:")
    print("  - Try modifying the examples with your own images and text")
    print("  - Explore different attributes and generation methods")
    print("  - Check out the CLI: python -m counterfactual_lab.cli --help")
    print("  - Review the documentation for advanced features")


if __name__ == "__main__":
    main()