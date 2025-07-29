"""Benchmark bias evaluation performance."""

import time
import statistics
from typing import Dict, List
import json
from pathlib import Path

from counterfactual_lab import BiasEvaluator


class EvaluationBenchmark:
    """Benchmark bias evaluation metrics."""
    
    def __init__(self):
        """Initialize evaluation benchmark."""
        self.results: Dict = {}
    
    def benchmark_metric(
        self, 
        metric: str, 
        dataset_sizes: List[int] = [100, 500, 1000],
        num_runs: int = 3
    ) -> Dict:
        """Benchmark a single evaluation metric."""
        print(f"📊 Benchmarking {metric} metric...")
        
        results = {}
        
        for size in dataset_sizes:
            print(f"  Dataset size: {size}")
            times = []
            
            for run in range(num_runs):
                # Simulate evaluation (replace with actual evaluation)
                start_time = time.time()
                
                # This would be replaced with actual metric computation
                time.sleep(0.01 * size / 100)  # Simulate work proportional to dataset size
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[size] = {
                "times": times,
                "avg_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "samples_per_second": size / statistics.mean(times)
            }
        
        return {
            "metric": metric,
            "results": results,
            "scalability": self._calculate_scalability(results)
        }
    
    def _calculate_scalability(self, results: Dict) -> Dict:
        """Calculate scalability metrics."""
        sizes = sorted(results.keys())
        if len(sizes) < 2:
            return {"complexity": "unknown"}
        
        # Calculate time complexity
        size_ratios = [sizes[i] / sizes[0] for i in range(len(sizes))]
        time_ratios = [results[sizes[i]]["avg_time"] / results[sizes[0]]["avg_time"] for i in range(len(sizes))]
        
        # Simple linear regression to estimate complexity
        if len(sizes) >= 3:
            # Check if it's linear (O(n)), quadratic (O(n²)), or other
            linear_fit = sum((tr - sr) ** 2 for tr, sr in zip(time_ratios, size_ratios))
            quadratic_fit = sum((tr - sr ** 2) ** 2 for tr, sr in zip(time_ratios, size_ratios))
            
            if linear_fit < quadratic_fit:
                complexity = "O(n) - Linear"
            else:
                complexity = "O(n²) - Quadratic"
        else:
            complexity = "Insufficient data"
        
        return {
            "complexity": complexity,
            "size_ratios": size_ratios,
            "time_ratios": time_ratios
        }
    
    def benchmark_all_metrics(self, **kwargs) -> Dict:
        """Benchmark all available evaluation metrics."""
        metrics = [
            "demographic_parity",
            "equalized_odds", 
            "cits_score",
            "disparate_impact",
            "statistical_parity_distance"
        ]
        
        results = {}
        
        for metric in metrics:
            try:
                results[metric] = self.benchmark_metric(metric, **kwargs)
            except Exception as e:
                print(f"❌ Failed to benchmark {metric}: {e}")
                results[metric] = {"error": str(e)}
        
        return results
    
    def analyze_scalability(self, results: Dict) -> Dict:
        """Analyze scalability across all metrics."""
        analysis = {
            "most_scalable": None,
            "least_scalable": None,
            "complexity_summary": {}
        }
        
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not valid_results:
            return analysis
        
        # Find most/least scalable based on largest dataset performance
        largest_size = max(
            max(result["results"].keys()) 
            for result in valid_results.values()
        )
        
        performance_at_scale = {}
        for metric, result in valid_results.items():
            if largest_size in result["results"]:
                performance_at_scale[metric] = result["results"][largest_size]["samples_per_second"]
        
        if performance_at_scale:
            analysis["most_scalable"] = max(performance_at_scale, key=performance_at_scale.get)
            analysis["least_scalable"] = min(performance_at_scale, key=performance_at_scale.get)
        
        # Complexity summary
        for metric, result in valid_results.items():
            analysis["complexity_summary"][metric] = result["scalability"]["complexity"]
        
        return analysis
    
    def save_results(self, results: Dict, filepath: str = "evaluation_benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results saved to {output_path}")
    
    def print_summary(self, results: Dict, analysis: Dict):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("📊 EVALUATION BENCHMARK SUMMARY")
        print("="*60)
        
        for metric, result in results.items():
            if "error" in result:
                print(f"\n❌ {metric.upper()}: {result['error']}")
                continue
            
            print(f"\n📈 {metric.upper()} PERFORMANCE:")
            print(f"  Complexity: {result['scalability']['complexity']}")
            
            # Show performance at different scales
            for size, perf in result["results"].items():
                print(f"  {size} samples: {perf['avg_time']:.3f}s ({perf['samples_per_second']:.1f} samples/sec)")
        
        if analysis["most_scalable"]:
            print(f"\n🚀 Most Scalable: {analysis['most_scalable'].upper()}")
            print(f"⚠️  Least Scalable: {analysis['least_scalable'].upper()}")
        
        print("\n📋 COMPLEXITY OVERVIEW:")
        for metric, complexity in analysis["complexity_summary"].items():
            print(f"  {metric}: {complexity}")
        
        print("\n" + "="*60)


def main():
    """Run evaluation benchmarks."""
    benchmark = EvaluationBenchmark()
    
    # Run benchmarks
    results = benchmark.benchmark_all_metrics(
        dataset_sizes=[50, 200, 500], 
        num_runs=3
    )
    analysis = benchmark.analyze_scalability(results)
    
    # Save and display results
    benchmark.save_results({
        "results": results,
        "analysis": analysis,
        "timestamp": time.time()
    })
    
    benchmark.print_summary(results, analysis)


if __name__ == "__main__":
    main()