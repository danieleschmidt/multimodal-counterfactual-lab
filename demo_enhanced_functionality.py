#!/usr/bin/env python3
"""
Enhanced functionality demonstration for Multimodal Counterfactual Lab
Showcases Generation 1 (Make it Work) improvements with autonomous execution
"""

import sys
import os
sys.path.insert(0, 'src')

from counterfactual_lab import CounterfactualGenerator, BiasEvaluator, _using_lightweight
import json
import time


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print('='*60)


def print_section(title: str):
    """Print section header."""
    print(f"\n🔹 {title}")
    print('-' * 40)


def demo_basic_generation():
    """Demonstrate basic counterfactual generation."""
    print_section("Basic Counterfactual Generation")
    
    # Initialize generator
    generator = CounterfactualGenerator(
        method="basic" if _using_lightweight else "modicf",
        device="cpu",
        use_cache=True
    )
    
    print(f"✅ Generator initialized (using {'lightweight' if _using_lightweight else 'full'} implementation)")
    
    # Generate counterfactuals
    result = generator.generate(
        text="A doctor examining a patient in the hospital",
        attributes=["gender", "age", "race"],
        num_samples=5
    )
    
    print(f"✅ Generated {len(result['counterfactuals'])} counterfactuals")
    print(f"⏱️  Generation time: {result['metadata']['generation_time']:.3f}s")
    print(f"💻 Device: {result['metadata']['device']}")
    
    # Show individual counterfactuals
    for i, cf in enumerate(result['counterfactuals'][:3]):  # Show first 3
        attrs = cf['target_attributes']
        confidence = cf.get('confidence', 'N/A')
        print(f"  CF{i+1}: {attrs} (confidence: {confidence:.3f})")
    
    return result


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print_section("Batch Processing")
    
    generator = CounterfactualGenerator(device="cpu")
    
    # Create multiple requests
    requests = [
        {
            "text": "A teacher explaining concepts to students",
            "attributes": ["gender", "age"],
            "num_samples": 2
        },
        {
            "text": "An engineer working on a technical project", 
            "attributes": ["race", "gender"],
            "num_samples": 2
        },
        {
            "text": "A scientist conducting research experiments",
            "attributes": ["age", "race"],
            "num_samples": 2
        }
    ]
    
    print(f"📦 Processing batch of {len(requests)} requests...")
    
    start_time = time.time()
    results = generator.generate_batch(requests)
    batch_time = time.time() - start_time
    
    total_cfs = sum(len(r['counterfactuals']) for r in results)
    
    print(f"✅ Batch processing completed")
    print(f"⏱️  Total time: {batch_time:.3f}s")
    print(f"📊 Total counterfactuals: {total_cfs}")
    print(f"📈 Average per request: {batch_time/len(requests):.3f}s")
    
    for i, result in enumerate(results):
        cf_count = len(result['counterfactuals'])
        text_preview = result['original_text'][:40] + "..."
        print(f"  Request {i+1}: {cf_count} CFs - {text_preview}")
    
    return results


def demo_bias_evaluation():
    """Demonstrate bias evaluation capabilities."""
    print_section("Bias Evaluation")
    
    # Mock VLM for evaluation
    class MockVisionLanguageModel:
        def __init__(self, name="demo-vlm-v1"):
            self.name = name
        
        def predict(self, image, text):
            # Mock prediction for demo
            import random
            return random.uniform(0.3, 0.9)
    
    # Initialize components
    generator = CounterfactualGenerator(device="cpu")
    model = MockVisionLanguageModel()
    evaluator = BiasEvaluator(model)
    
    print(f"🤖 Mock VLM: {model.name}")
    
    # Generate counterfactuals for evaluation
    cf_data = generator.generate(
        text="A professional presenting quarterly results",
        attributes=["gender", "race", "age"],
        num_samples=6
    )
    
    print(f"📊 Generated {len(cf_data['counterfactuals'])} counterfactuals for evaluation")
    
    # Evaluate bias
    metrics = ["demographic_parity", "attribute_balance", "confidence_distribution"]
    if not _using_lightweight:
        metrics.extend(["equalized_odds", "cits_score"])
    
    evaluation_results = evaluator.evaluate(cf_data, metrics)
    
    print(f"✅ Bias evaluation completed")
    
    # Show results
    summary = evaluation_results['summary']
    print(f"🏆 Overall fairness score: {summary['overall_fairness_score']:.3f}")
    print(f"📋 Rating: {summary['fairness_rating']}")
    print(f"📈 Metrics evaluated: {len(evaluation_results['metrics'])}")
    
    # Show individual metric results
    for metric, data in evaluation_results['metrics'].items():
        if isinstance(data, dict):
            if 'error' in data:
                print(f"  ❌ {metric}: {data['error']}")
            elif 'overall_balance_score' in data:
                score = data['overall_balance_score']
                print(f"  📊 {metric}: {score:.3f}")
            elif 'overall_balance' in data:
                score = data['overall_balance']
                print(f"  📊 {metric}: {score:.3f}")
            elif 'quality_assessment' in data:
                quality = data['quality_assessment']
                print(f"  🔍 {metric}: {quality}")
    
    return evaluation_results


def demo_caching_and_storage():
    """Demonstrate caching and storage features."""
    print_section("Caching & Storage")
    
    import tempfile
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = f"{temp_dir}/cache"
        storage_dir = f"{temp_dir}/storage"
        
        generator = CounterfactualGenerator(
            device="cpu",
            use_cache=True,
            cache_dir=cache_dir,
            storage_dir=storage_dir
        )
        
        print(f"📁 Cache dir: {cache_dir}")
        print(f"💾 Storage dir: {storage_dir}")
        
        # First generation (cache miss)
        print("🔄 First generation (cache miss)...")
        result1 = generator.generate(
            text="A researcher analyzing data patterns",
            attributes=["gender", "age"], 
            num_samples=3,
            save_results=True,
            experiment_id="demo_caching_001"
        )
        
        gen_time_1 = result1['metadata']['generation_time']
        print(f"⏱️  Generation time: {gen_time_1:.3f}s")
        
        # Second identical generation (cache hit)
        print("🔄 Second generation (cache hit)...")
        result2 = generator.generate(
            text="A researcher analyzing data patterns",
            attributes=["gender", "age"],
            num_samples=3
        )
        
        gen_time_2 = result2['metadata']['generation_time']
        cache_hit = result2['metadata'].get('cache_hit', False)
        
        print(f"⏱️  Generation time: {gen_time_2:.3f}s")
        print(f"🎯 Cache hit: {cache_hit}")
        print(f"⚡ Speedup: {gen_time_1/gen_time_2:.1f}x" if gen_time_2 > 0 else "N/A")
        
        # Show storage info
        if "saved_path" in result1['metadata']:
            print(f"💾 Results saved to: {result1['metadata']['saved_path']}")
        
        # System status
        status = generator.get_system_status()
        print(f"📈 Total generations: {status['statistics']['total_generations']}")
        print(f"🗃️  Cache entries: {status['statistics']['cache_entries']}")
    
    return result1, result2


def demo_advanced_attributes():
    """Demonstrate advanced attribute handling."""
    print_section("Advanced Attribute Handling")
    
    generator = CounterfactualGenerator(device="cpu")
    
    # Test various attribute combinations
    test_cases = [
        {
            "text": "A professional in business attire",
            "attributes": ["gender", "age", "profession"],
            "description": "Multi-attribute generation"
        },
        {
            "text": "A person giving a presentation",
            "attributes": ["race", "setting", "expression"],
            "description": "Context-aware attributes"
        },
        {
            "text": "An individual working with technology",
            "attributes": ["hair", "clothing", "age"],
            "description": "Physical appearance attributes"
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"🧪 Test {i+1}: {test_case['description']}")
        
        result = generator.generate(
            text=test_case['text'],
            attributes=test_case['attributes'],
            num_samples=3
        )
        
        print(f"  ✅ Generated {len(result['counterfactuals'])} counterfactuals")
        
        # Show attribute variety
        all_attrs = set()
        for cf in result['counterfactuals']:
            all_attrs.update(cf['target_attributes'].keys())
        
        print(f"  📊 Attributes covered: {sorted(all_attrs)}")
        
        # Show sample
        sample_cf = result['counterfactuals'][0]
        sample_attrs = sample_cf['target_attributes']
        print(f"  🎯 Sample: {sample_attrs}")
        
        results.append(result)
    
    return results


def demo_system_monitoring():
    """Demonstrate system monitoring capabilities."""
    print_section("System Monitoring")
    
    generator = CounterfactualGenerator(device="cpu")
    
    # Generate some data for monitoring
    for i in range(3):
        generator.generate(
            text=f"Sample generation {i+1}",
            attributes=["gender"],
            num_samples=1
        )
    
    # Get system status
    status = generator.get_system_status()
    
    print("🖥️  System Status:")
    gen_info = status['generator']
    print(f"  Method: {gen_info['method']}")
    print(f"  Device: {gen_info['device']}")
    print(f"  Cache enabled: {gen_info['cache_enabled']}")
    
    stats = status['statistics']
    print(f"  Total generations: {stats['total_generations']}")
    print(f"  Cache entries: {stats['cache_entries']}")
    
    dirs = status['directories']
    print(f"  Cache directory: {dirs['cache_dir']}")
    print(f"  Storage directory: {dirs['storage_dir']}")
    
    # Cleanup resources
    print("🧹 Cleaning up resources...")
    generator.cleanup_resources()
    print("✅ Cleanup completed")
    
    return status


def demo_report_generation():
    """Demonstrate report generation."""
    print_section("Report Generation")
    
    # Generate evaluation data
    generator = CounterfactualGenerator(device="cpu")
    evaluator = BiasEvaluator({"name": "demo_model"})
    
    cf_data = generator.generate(
        text="A healthcare professional consulting with patients",
        attributes=["gender", "race"],
        num_samples=4
    )
    
    evaluation_results = evaluator.evaluate(
        cf_data, 
        ["demographic_parity", "attribute_balance"]
    )
    
    # Generate report
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_path = f.name
    
    report = evaluator.generate_report(
        evaluation_results,
        format="simple",
        export_path=report_path
    )
    
    print(f"📄 Report generated: {report_path}")
    print(f"📊 Report type: {report['report_type']}")
    print(f"🏆 Overall score: {report['summary']['overall_fairness_score']:.3f}")
    print(f"📋 Key findings: {len(report['key_findings'])} items")
    
    for finding in report['key_findings']:
        print(f"  • {finding}")
    
    # Cleanup
    os.unlink(report_path)
    
    return report


def run_comprehensive_demo():
    """Run comprehensive demonstration of all features."""
    print_header("GENERATION 1: ENHANCED FUNCTIONALITY DEMONSTRATION")
    
    if _using_lightweight:
        print("ℹ️  Running with lightweight implementation (no heavy dependencies)")
    else:
        print("ℹ️  Running with full implementation")
    
    # Track results for final summary
    demo_results = {}
    
    try:
        demo_results['basic'] = demo_basic_generation()
        demo_results['batch'] = demo_batch_processing()
        demo_results['bias_eval'] = demo_bias_evaluation()
        demo_results['caching'] = demo_caching_and_storage()
        demo_results['attributes'] = demo_advanced_attributes()
        demo_results['monitoring'] = demo_system_monitoring()
        demo_results['reporting'] = demo_report_generation()
        
        print_header("🎉 DEMONSTRATION SUMMARY")
        
        # Collect stats
        total_counterfactuals = 0
        total_experiments = len(demo_results)
        
        for key, result in demo_results.items():
            if key == 'batch':
                total_counterfactuals += sum(len(r['counterfactuals']) for r in result)
            elif key in ['basic', 'caching']:
                total_counterfactuals += len(result[0]['counterfactuals']) if isinstance(result, tuple) else len(result['counterfactuals'])
            elif key == 'attributes':
                total_counterfactuals += sum(len(r['counterfactuals']) for r in result)
        
        print(f"✅ Successfully completed {total_experiments} demonstration modules")
        print(f"📊 Total counterfactuals generated: {total_counterfactuals}")
        print(f"🧪 Bias evaluations performed: 2")
        print(f"📄 Reports generated: 1")
        print(f"🚀 Implementation: {'Lightweight' if _using_lightweight else 'Full-featured'}")
        
        print(f"\n🎯 Generation 1 (Make it Work) - STATUS: ✅ COMPLETED")
        print("   ├─ Basic counterfactual generation ✅")
        print("   ├─ Batch processing capabilities ✅")
        print("   ├─ Bias evaluation framework ✅") 
        print("   ├─ Caching and storage system ✅")
        print("   ├─ Advanced attribute handling ✅")
        print("   ├─ System monitoring ✅")
        print("   └─ Report generation ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_demo()
    exit(0 if success else 1)