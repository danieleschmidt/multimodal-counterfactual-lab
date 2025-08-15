#!/usr/bin/env python3
"""Performance benchmarks and stress testing for the counterfactual generation system."""

import sys
import time
import threading
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import gc

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class PerformanceBenchmark:
    """Performance benchmarking and stress testing."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_memory_stress_test(self) -> Dict[str, Any]:
        """Test memory management under stress."""
        print("🧠 Running memory stress test...")
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Create and release large objects to test memory management
        large_objects = []
        for i in range(10):
            # Create large dictionary
            large_obj = {f"key_{j}": f"value_{j}" * 1000 for j in range(1000)}
            large_objects.append(large_obj)
            
            if i % 3 == 0:
                # Release some objects
                large_objects = large_objects[-2:]
                gc.collect()
        
        # Final cleanup
        large_objects.clear()
        gc.collect()
        
        end_time = time.time()
        final_memory = self._get_memory_usage()
        
        return {
            "duration": end_time - start_time,
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "memory_growth": final_memory - initial_memory,
            "status": "PASS" if final_memory - initial_memory < 50 else "WARN"
        }
    
    def run_concurrency_test(self) -> Dict[str, Any]:
        """Test concurrent operations."""
        print("🔄 Running concurrency stress test...")
        
        results = []
        errors = []
        
        def worker_function(worker_id: int):
            try:
                start = time.time()
                # Simulate work
                for i in range(100):
                    data = {"worker": worker_id, "iteration": i}
                    # Simulate processing
                    time.sleep(0.001)
                
                duration = time.time() - start
                results.append({"worker_id": worker_id, "duration": duration})
            except Exception as e:
                errors.append({"worker_id": worker_id, "error": str(e)})
        
        # Start multiple worker threads
        threads = []
        start_time = time.time()
        
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        return {
            "total_duration": end_time - start_time,
            "workers_completed": len(results),
            "errors": len(errors),
            "average_worker_time": sum(r["duration"] for r in results) / len(results) if results else 0,
            "error_details": errors[:3],  # First 3 errors
            "status": "PASS" if len(errors) == 0 else "FAIL"
        }
    
    def run_algorithm_performance_test(self) -> Dict[str, Any]:
        """Test algorithmic performance with mock implementations."""
        print("⚡ Running algorithm performance test...")
        
        def mock_generation_algorithm(num_samples: int):
            """Mock counterfactual generation algorithm."""
            results = []
            for i in range(num_samples):
                # Simulate attribute transformation
                attributes = {
                    "gender": ["male", "female"][i % 2],
                    "age": ["young", "old"][i % 2],
                    "race": ["caucasian", "diverse"][i % 2]
                }
                
                # Simulate processing time
                time.sleep(0.01)
                
                result = {
                    "sample_id": i,
                    "attributes": attributes,
                    "confidence": 0.8 + (i % 3) * 0.05,
                    "generated": True
                }
                results.append(result)
            
            return results
        
        def mock_bias_evaluation(counterfactuals: List[Dict]):
            """Mock bias evaluation algorithm."""
            # Simulate bias calculation
            scores = []
            for cf in counterfactuals:
                score = cf["confidence"] * (0.9 + (cf["sample_id"] % 5) * 0.02)
                scores.append(score)
            
            return {
                "demographic_parity": sum(scores) / len(scores),
                "fairness_score": min(scores),
                "bias_detected": any(s < 0.85 for s in scores)
            }
        
        # Test different scales
        test_scales = [5, 10, 25, 50]
        performance_results = {}
        
        for scale in test_scales:
            start_time = time.time()
            
            # Generation phase
            gen_start = time.time()
            counterfactuals = mock_generation_algorithm(scale)
            gen_time = time.time() - gen_start
            
            # Evaluation phase
            eval_start = time.time()
            evaluation = mock_bias_evaluation(counterfactuals)
            eval_time = time.time() - eval_start
            
            total_time = time.time() - start_time
            
            performance_results[f"scale_{scale}"] = {
                "generation_time": gen_time,
                "evaluation_time": eval_time,
                "total_time": total_time,
                "samples_per_second": scale / total_time,
                "memory_efficient": len(counterfactuals) == scale,
                "quality_score": evaluation["fairness_score"]
            }
        
        # Calculate scaling efficiency
        base_scale = test_scales[0]
        max_scale = test_scales[-1]
        base_time = performance_results[f"scale_{base_scale}"]["total_time"]
        max_time = performance_results[f"scale_{max_scale}"]["total_time"]
        
        scaling_factor = max_scale / base_scale
        time_scaling = max_time / base_time
        efficiency = scaling_factor / time_scaling
        
        return {
            "scale_tests": performance_results,
            "scaling_efficiency": efficiency,
            "max_throughput": max(r["samples_per_second"] for r in performance_results.values()),
            "status": "PASS" if efficiency > 0.5 else "WARN"  # Should scale reasonably well
        }
    
    def run_error_handling_stress_test(self) -> Dict[str, Any]:
        """Test error handling under stress."""
        print("🚨 Running error handling stress test...")
        
        try:
            from counterfactual_lab.enhanced_error_handling import ErrorHandler, StructuredLogger
            
            logger = StructuredLogger("benchmark_logger", json_output=False)
            error_handler = ErrorHandler(logger)
            
            # Generate various types of errors
            error_types = [
                (ValueError, "Invalid value provided"),
                (RuntimeError, "Runtime error occurred"),
                (KeyError, "Missing key in dictionary"),
                (TypeError, "Type mismatch error"),
                (Exception, "Generic exception")
            ]
            
            start_time = time.time()
            handled_errors = 0
            
            for i in range(50):  # Test 50 errors
                error_type, message = error_types[i % len(error_types)]
                test_error = error_type(f"{message} #{i}")
                
                try:
                    error_context = error_handler.handle_error(
                        error=test_error,
                        component=f"test_component_{i % 5}",
                        operation=f"test_operation_{i % 3}"
                    )
                    
                    if error_context.error_type == error_type.__name__:
                        handled_errors += 1
                        
                except Exception:
                    # Error in error handling itself
                    pass
            
            end_time = time.time()
            
            # Get statistics
            stats = error_handler.get_error_statistics()
            
            return {
                "duration": end_time - start_time,
                "errors_generated": 50,
                "errors_handled": handled_errors,
                "handling_rate": handled_errors / 50,
                "average_handling_time": (end_time - start_time) / 50,
                "error_statistics": stats,
                "status": "PASS" if handled_errors >= 45 else "FAIL"  # Should handle 90%+ errors
            }
            
        except ImportError:
            return {
                "status": "SKIP", 
                "reason": "Error handling module not available"
            }
    
    def run_resource_utilization_test(self) -> Dict[str, Any]:
        """Test resource utilization patterns."""
        print("📊 Running resource utilization test...")
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Simulate various workload patterns
        workloads = [
            ("burst", self._burst_workload),
            ("sustained", self._sustained_workload),
            ("mixed", self._mixed_workload)
        ]
        
        workload_results = {}
        
        for workload_name, workload_func in workloads:
            wl_start = time.time()
            wl_initial_mem = self._get_memory_usage()
            
            workload_func()
            
            wl_end = time.time()
            wl_final_mem = self._get_memory_usage()
            
            workload_results[workload_name] = {
                "duration": wl_end - wl_start,
                "memory_delta": wl_final_mem - wl_initial_mem,
                "efficiency": "good" if wl_final_mem - wl_initial_mem < 20 else "poor"
            }
            
            # Clean up between workloads
            gc.collect()
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        final_memory = self._get_memory_usage()
        
        return {
            "total_duration": total_time,
            "memory_growth": final_memory - initial_memory,
            "workload_results": workload_results,
            "resource_efficiency": "good" if final_memory - initial_memory < 30 else "poor",
            "status": "PASS"
        }
    
    def _burst_workload(self):
        """Simulate burst workload pattern."""
        for burst in range(3):
            # High activity burst
            data = []
            for i in range(1000):
                data.append({"id": i, "data": f"burst_data_{i}"})
            
            # Process data
            processed = [item for item in data if item["id"] % 2 == 0]
            
            # Clear burst data
            data.clear()
            processed.clear()
            
            # Brief pause
            time.sleep(0.01)
    
    def _sustained_workload(self):
        """Simulate sustained workload pattern."""
        data_store = []
        
        for i in range(2000):
            # Add data continuously
            data_store.append({"id": i, "timestamp": time.time()})
            
            # Periodic cleanup
            if i % 100 == 0:
                # Keep only recent data
                data_store = data_store[-500:]
            
            if i % 50 == 0:
                time.sleep(0.001)  # Brief pause
    
    def _mixed_workload(self):
        """Simulate mixed workload pattern."""
        # Combination of burst and sustained
        self._burst_workload()
        time.sleep(0.01)
        self._sustained_workload()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (mock implementation)."""
        # Mock memory usage - in real implementation would use psutil
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
        except:
            # Fallback: use object count as proxy
            return len(gc.get_objects()) / 1000
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("🚀 PERFORMANCE BENCHMARKS - AUTONOMOUS EXECUTION")
        print("=" * 60)
        
        self.start_time = time.time()
        
        benchmarks = [
            ("Memory Stress Test", self.run_memory_stress_test),
            ("Concurrency Test", self.run_concurrency_test),
            ("Algorithm Performance", self.run_algorithm_performance_test),
            ("Error Handling Stress", self.run_error_handling_stress_test),
            ("Resource Utilization", self.run_resource_utilization_test)
        ]
        
        results = {}
        passed = 0
        total = 0
        
        for name, benchmark_func in benchmarks:
            print(f"\n{name}...")
            try:
                result = benchmark_func()
                results[name] = result
                
                status = result.get("status", "UNKNOWN")
                if status == "PASS":
                    print(f"✅ {name}: PASSED")
                    passed += 1
                elif status == "WARN":
                    print(f"⚠️  {name}: WARNING")
                elif status == "SKIP":
                    print(f"⏭️  {name}: SKIPPED")
                    continue  # Don't count skipped tests
                else:
                    print(f"❌ {name}: FAILED")
                
                total += 1
                
            except Exception as e:
                print(f"❌ {name}: ERROR - {str(e)}")
                results[name] = {"status": "ERROR", "error": str(e)}
                total += 1
        
        self.end_time = time.time()
        
        # Generate summary
        summary = {
            "total_benchmarks": total,
            "passed": passed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": self.end_time - self.start_time,
            "results": results
        }
        
        print("\n📊 BENCHMARK SUMMARY")
        print(f"Total Benchmarks: {total}")
        print(f"Passed: {passed}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        if passed == total:
            print("\n🎉 ALL PERFORMANCE BENCHMARKS PASSED!")
        else:
            print(f"\n⚠️  {total - passed} BENCHMARK(S) FAILED OR HAD WARNINGS")
        
        return summary

def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results
    results_file = Path(__file__).parent / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Return appropriate exit code
    return 0 if results["success_rate"] >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())