"""Benchmark counterfactual generation performance."""

import time
from typing import Dict, List
import statistics
import psutil
import torch
from pathlib import Path
import json

from counterfactual_lab import CounterfactualGenerator


class GenerationBenchmark:
    """Benchmark counterfactual generation methods."""
    
    def __init__(self, device: str = "auto"):
        """Initialize benchmark suite."""
        self.device = self._get_device(device)
        self.results: Dict = {}
        
    def _get_device(self, device: str) -> str:
        """Determine optimal device for benchmarking."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def benchmark_method(
        self, 
        method: str, 
        num_samples: int = 10,
        num_runs: int = 3,
        image_size: tuple = (512, 512)
    ) -> Dict:
        """Benchmark a single generation method."""
        print(f"🏃 Benchmarking {method} method...")
        
        generator = CounterfactualGenerator(method=method, device=self.device)
        
        times = []
        memory_usage = []
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            # Monitor memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Time generation
            start_time = time.time()
            
            # Simulate generation (replace with actual generation)
            for _ in range(num_samples):
                # This would be replaced with actual generation call
                time.sleep(0.1)  # Simulate work
            
            end_time = time.time()
            
            # Monitor memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            if torch.cuda.is_available():
                gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                memory_used = max(memory_used, gpu_memory_peak)
            
            times.append(end_time - start_time)
            memory_usage.append(memory_used)
        
        return {
            "method": method,
            "num_samples": num_samples,
            "num_runs": num_runs,
            "times": times,
            "avg_time": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "memory_usage": memory_usage,
            "avg_memory": statistics.mean(memory_usage),
            "samples_per_second": num_samples / statistics.mean(times),
            "device": self.device
        }
    
    def benchmark_all_methods(self, **kwargs) -> Dict:
        """Benchmark all available generation methods."""
        methods = ["modicf", "icg"]
        results = {}
        
        for method in methods:
            try:
                results[method] = self.benchmark_method(method, **kwargs)
            except Exception as e:
                print(f"❌ Failed to benchmark {method}: {e}")
                results[method] = {"error": str(e)}
        
        return results
    
    def compare_methods(self, results: Dict) -> Dict:
        """Compare performance across methods."""
        comparison = {
            "fastest_method": None,
            "most_memory_efficient": None,
            "performance_ratios": {}
        }
        
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not valid_results:
            return comparison
        
        # Find fastest method
        fastest = min(valid_results.items(), key=lambda x: x[1]["avg_time"])
        comparison["fastest_method"] = fastest[0]
        
        # Find most memory efficient
        most_efficient = min(valid_results.items(), key=lambda x: x[1]["avg_memory"])
        comparison["most_memory_efficient"] = most_efficient[0]
        
        # Calculate performance ratios
        baseline_time = fastest[1]["avg_time"]
        baseline_memory = most_efficient[1]["avg_memory"]
        
        for method, result in valid_results.items():
            comparison["performance_ratios"][method] = {
                "time_ratio": result["avg_time"] / baseline_time,
                "memory_ratio": result["avg_memory"] / baseline_memory,
                "throughput": result["samples_per_second"]
            }
        
        return comparison
    
    def save_results(self, results: Dict, filepath: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results saved to {output_path}")
    
    def print_summary(self, results: Dict, comparison: Dict):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("🏆 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        for method, result in results.items():
            if "error" in result:
                print(f"\n❌ {method.upper()}: {result['error']}")
                continue
                
            print(f"\n📈 {method.upper()} PERFORMANCE:")
            print(f"  Average Time: {result['avg_time']:.3f}s (±{result['std_time']:.3f}s)")
            print(f"  Memory Usage: {result['avg_memory']:.1f} MB")
            print(f"  Throughput: {result['samples_per_second']:.2f} samples/sec")
        
        if comparison["fastest_method"]:
            print(f"\n🚀 Fastest Method: {comparison['fastest_method'].upper()}")
            print(f"💾 Most Memory Efficient: {comparison['most_memory_efficient'].upper()}")
        
        print("\n" + "="*60)


def main():
    """Run performance benchmarks."""
    benchmark = GenerationBenchmark()
    
    # Run benchmarks
    results = benchmark.benchmark_all_methods(num_samples=5, num_runs=3)
    comparison = benchmark.compare_methods(results)
    
    # Save and display results
    benchmark.save_results({
        "results": results,
        "comparison": comparison,
        "timestamp": time.time(),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            "gpu_available": torch.cuda.is_available(),
            "gpu_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None
        }
    })
    
    benchmark.print_summary(results, comparison)


if __name__ == "__main__":
    main()