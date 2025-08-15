"""Tests for auto-scaling and performance optimization functionality."""

import pytest
import threading
import time
import queue
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from counterfactual_lab.auto_scaling import (
    WorkerPool,
    AdaptiveLoadBalancer,
    ScalingConfig,
    LoadMetrics,
    get_global_load_balancer,
    initialize_auto_scaling
)


class TestScalingConfig:
    """Test scaling configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ScalingConfig()
        
        assert config.min_workers == 1
        assert config.max_workers == 8
        assert config.target_cpu_utilization == 70.0
        assert config.target_memory_utilization == 80.0
        assert config.scale_up_threshold == 85.0
        assert config.scale_down_threshold == 50.0
        assert config.scale_up_cooldown == 60
        assert config.scale_down_cooldown == 300
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScalingConfig(
            min_workers=2,
            max_workers=16,
            target_cpu_utilization=60.0,
            scale_up_threshold=75.0
        )
        
        assert config.min_workers == 2
        assert config.max_workers == 16
        assert config.target_cpu_utilization == 60.0
        assert config.scale_up_threshold == 75.0


class TestLoadMetrics:
    """Test load metrics data structure."""
    
    def test_load_metrics_creation(self):
        """Test creating load metrics."""
        metrics = LoadMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_utilization=75.0,
            memory_utilization=65.0,
            gpu_utilization=80.0,
            queue_depth=5,
            active_workers=3,
            requests_per_second=10.5,
            average_response_time=2.5,
            error_rate=1.2
        )
        
        assert metrics.cpu_utilization == 75.0
        assert metrics.memory_utilization == 65.0
        assert metrics.gpu_utilization == 80.0
        assert metrics.queue_depth == 5
        assert metrics.active_workers == 3
        assert metrics.requests_per_second == 10.5
        assert metrics.average_response_time == 2.5
        assert metrics.error_rate == 1.2


class TestWorkerPool:
    """Test worker pool functionality."""
    
    @pytest.fixture
    def simple_worker_function(self):
        """Simple worker function for testing."""
        def worker_func(*args, **kwargs):
            # Simulate some work
            time.sleep(0.01)
            return {"result": "success", "args": args, "kwargs": kwargs}
        return worker_func
    
    @pytest.fixture
    def worker_pool(self, simple_worker_function):
        """Create test worker pool."""
        config = ScalingConfig(
            min_workers=1,
            max_workers=4,
            scale_up_cooldown=1,  # Short cooldown for testing
            scale_down_cooldown=2
        )
        pool = WorkerPool(
            worker_function=simple_worker_function,
            scaling_config=config,
            initial_workers=1
        )
        yield pool
        pool.shutdown()
    
    def test_worker_pool_initialization(self, worker_pool):
        """Test worker pool initialization."""
        assert worker_pool.current_workers >= 1
        assert worker_pool.target_workers >= 1
        assert len(worker_pool.workers) >= 1
        assert worker_pool.config.min_workers == 1
        assert worker_pool.config.max_workers == 4
    
    def test_task_submission_and_execution(self, worker_pool):
        """Test task submission and execution."""
        # Submit a task
        success = worker_pool.submit_task("test_arg", test_kwarg="test_value")
        assert success is True
        
        # Get result
        result = worker_pool.get_result(timeout=2)
        assert result is not None
        assert result["success"] is True
        assert result["result"]["args"] == ("test_arg",)
        assert result["result"]["kwargs"] == {"test_kwarg": "test_value"}
    
    def test_multiple_tasks(self, worker_pool):
        """Test handling multiple tasks."""
        num_tasks = 5
        
        # Submit multiple tasks
        for i in range(num_tasks):
            success = worker_pool.submit_task(f"task_{i}")
            assert success is True
        
        # Collect results
        results = []
        for i in range(num_tasks):
            result = worker_pool.get_result(timeout=5)
            assert result is not None
            assert result["success"] is True
            results.append(result)
        
        assert len(results) == num_tasks
    
    def test_error_handling(self, worker_pool):
        """Test error handling in worker pool."""
        def failing_worker(*args, **kwargs):
            raise Exception("Test error")
        
        # Replace worker function temporarily
        original_func = worker_pool.worker_function
        worker_pool.worker_function = failing_worker
        
        try:
            # Submit task that will fail
            success = worker_pool.submit_task("test")
            assert success is True
            
            # Get error result
            result = worker_pool.get_result(timeout=2)
            assert result is not None
            assert result["success"] is False
            assert "Test error" in result["error"]
            
        finally:
            # Restore original function
            worker_pool.worker_function = original_func
    
    def test_load_metrics_collection(self, worker_pool):
        """Test load metrics collection."""
        metrics = worker_pool.get_load_metrics()
        
        assert isinstance(metrics, LoadMetrics)
        assert metrics.cpu_utilization >= 0
        assert metrics.memory_utilization >= 0
        assert metrics.active_workers >= 0
        assert metrics.queue_depth >= 0
        assert metrics.requests_per_second >= 0
        assert metrics.average_response_time >= 0
        assert metrics.error_rate >= 0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_scaling_decisions(self, mock_memory, mock_cpu, simple_worker_function):
        """Test auto-scaling decisions."""
        # Mock high resource usage
        mock_cpu.return_value = 95.0  # High CPU
        mock_memory.return_value.percent = 90.0  # High memory
        
        config = ScalingConfig(
            min_workers=1,
            max_workers=4,
            scale_up_threshold=85.0,
            scale_up_cooldown=0  # No cooldown for testing
        )
        
        pool = WorkerPool(simple_worker_function, config, initial_workers=1)
        
        try:
            # Get metrics with high resource usage
            metrics = pool.get_load_metrics()
            
            # Should decide to scale up
            should_scale_up = pool.should_scale_up(metrics)
            assert should_scale_up is True
            
            # Perform auto-scaling
            action_taken = pool.auto_scale()
            assert action_taken is True
            assert pool.target_workers > 1
            
        finally:
            pool.shutdown()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_scale_down_decision(self, mock_memory, mock_cpu, simple_worker_function):
        """Test scale down decisions."""
        # Mock low resource usage
        mock_cpu.return_value = 30.0  # Low CPU
        mock_memory.return_value.percent = 40.0  # Low memory
        
        config = ScalingConfig(
            min_workers=1,
            max_workers=4,
            scale_down_threshold=50.0,
            scale_down_cooldown=0  # No cooldown for testing
        )
        
        pool = WorkerPool(simple_worker_function, config, initial_workers=3)
        
        try:
            # Get metrics with low resource usage
            metrics = pool.get_load_metrics()
            
            # Should decide to scale down
            should_scale_down = pool.should_scale_down(metrics)
            assert should_scale_down is True
            
            # Perform auto-scaling
            action_taken = pool.auto_scale()
            assert action_taken is True
            assert pool.target_workers < 3
            
        finally:
            pool.shutdown()
    
    def test_cooldown_prevents_rapid_scaling(self, simple_worker_function):
        """Test that cooldown prevents rapid scaling."""
        config = ScalingConfig(
            min_workers=1,
            max_workers=4,
            scale_up_cooldown=60,  # Long cooldown
            scale_up_threshold=0.0  # Always trigger scale up
        )
        
        pool = WorkerPool(simple_worker_function, config, initial_workers=1)
        
        try:
            # First scale up should work
            action_taken = pool.auto_scale()
            initial_workers = pool.target_workers
            
            # Second scale up should be blocked by cooldown
            action_taken = pool.auto_scale()
            assert pool.target_workers == initial_workers  # No change
            
        finally:
            pool.shutdown()
    
    def test_worker_statistics(self, worker_pool):
        """Test worker pool statistics."""
        # Submit and complete some tasks
        for i in range(3):
            worker_pool.submit_task(f"task_{i}")
        
        # Wait for tasks to complete
        for i in range(3):
            worker_pool.get_result(timeout=2)
        
        stats = worker_pool.get_statistics()
        
        assert "current_workers" in stats
        assert "target_workers" in stats
        assert "total_requests" in stats
        assert "error_count" in stats
        assert "error_rate" in stats
        assert "queue_depth" in stats
        assert "worker_stats" in stats
        assert "scaling_config" in stats
        
        assert stats["total_requests"] >= 3


class TestAdaptiveLoadBalancer:
    """Test adaptive load balancer."""
    
    @pytest.fixture
    def simple_worker_function(self):
        """Simple worker function for testing."""
        def worker_func(task_type, *args, **kwargs):
            time.sleep(0.01)
            return {"task_type": task_type, "result": "success"}
        return worker_func
    
    @pytest.fixture
    def load_balancer(self, simple_worker_function):
        """Create test load balancer."""
        scaling_configs = {
            "test_pool": ScalingConfig(min_workers=1, max_workers=3)
        }
        
        balancer = AdaptiveLoadBalancer(
            scaling_configs=scaling_configs,
            load_balancing_strategy="least_loaded"
        )
        
        # Register a test worker pool
        balancer.register_worker_pool(
            pool_name="test_pool",
            worker_function=simple_worker_function,
            task_types=["test_task", "another_task"]
        )
        
        yield balancer
        balancer.shutdown()
    
    def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization."""
        assert len(load_balancer.worker_pools) == 1
        assert "test_pool" in load_balancer.worker_pools
        assert load_balancer.task_routing["test_task"] == "test_pool"
        assert load_balancer.task_routing["another_task"] == "test_pool"
        assert load_balancer.strategy == "least_loaded"
    
    def test_task_submission(self, load_balancer):
        """Test task submission to load balancer."""
        # Submit a task
        pool_name = load_balancer.submit_task("test_task", "arg1", kwarg1="value1")
        assert pool_name == "test_pool"
        
        # Get result from the pool
        pool = load_balancer.worker_pools[pool_name]
        result = pool.get_result(timeout=2)
        
        assert result is not None
        assert result["success"] is True
        assert result["result"]["task_type"] == "test_task"
    
    def test_unknown_task_type(self, load_balancer):
        """Test handling of unknown task types."""
        pool_name = load_balancer.submit_task("unknown_task")
        assert pool_name is None
    
    def test_monitoring_start_stop(self, load_balancer):
        """Test monitoring start and stop."""
        assert not load_balancer.monitoring
        
        # Start monitoring
        load_balancer.start_monitoring(interval=1)
        assert load_balancer.monitoring is True
        assert load_balancer.monitor_thread is not None
        
        # Stop monitoring
        load_balancer.stop_monitoring()
        assert load_balancer.monitoring is False
    
    def test_global_statistics(self, load_balancer):
        """Test global statistics collection."""
        # Submit some tasks
        for i in range(3):
            load_balancer.submit_task("test_task", f"arg_{i}")
        
        # Wait for completion
        pool = load_balancer.worker_pools["test_pool"]
        for i in range(3):
            pool.get_result(timeout=2)
        
        stats = load_balancer.get_global_statistics()
        
        assert "total_pools" in stats
        assert "total_workers" in stats
        assert "total_requests" in stats
        assert "global_error_rate" in stats
        assert "pool_statistics" in stats
        assert "task_routing" in stats
        assert "monitoring_active" in stats
        
        assert stats["total_pools"] == 1
        assert stats["total_requests"] >= 3
    
    def test_multiple_pools(self):
        """Test load balancer with multiple pools."""
        def worker_func_1(task_type, *args, **kwargs):
            return {"pool": "pool1", "task_type": task_type}
        
        def worker_func_2(task_type, *args, **kwargs):
            return {"pool": "pool2", "task_type": task_type}
        
        scaling_configs = {
            "pool1": ScalingConfig(min_workers=1, max_workers=2),
            "pool2": ScalingConfig(min_workers=1, max_workers=2)
        }
        
        balancer = AdaptiveLoadBalancer(scaling_configs)
        
        try:
            # Register two pools
            balancer.register_worker_pool("pool1", worker_func_1, ["task_type_1"])
            balancer.register_worker_pool("pool2", worker_func_2, ["task_type_2"])
            
            # Submit tasks to different pools
            pool1_name = balancer.submit_task("task_type_1")
            pool2_name = balancer.submit_task("task_type_2")
            
            assert pool1_name == "pool1"
            assert pool2_name == "pool2"
            
            # Verify results
            result1 = balancer.worker_pools["pool1"].get_result(timeout=2)
            result2 = balancer.worker_pools["pool2"].get_result(timeout=2)
            
            assert result1["result"]["pool"] == "pool1"
            assert result2["result"]["pool"] == "pool2"
            
        finally:
            balancer.shutdown()


class TestGlobalFunctions:
    """Test global auto-scaling functions."""
    
    def test_get_global_load_balancer(self):
        """Test getting global load balancer."""
        # Clear any existing global balancer
        import counterfactual_lab.auto_scaling as auto_scaling
        auto_scaling._global_load_balancer = None
        
        balancer1 = get_global_load_balancer()
        balancer2 = get_global_load_balancer()
        
        # Should return same instance
        assert balancer1 is balancer2
        assert isinstance(balancer1, AdaptiveLoadBalancer)
        
        # Cleanup
        balancer1.shutdown()
    
    def test_initialize_auto_scaling(self):
        """Test auto-scaling initialization."""
        # Clear any existing global balancer
        import counterfactual_lab.auto_scaling as auto_scaling
        auto_scaling._global_load_balancer = None
        
        scaling_configs = {
            "test_pool": ScalingConfig(min_workers=2, max_workers=6)
        }
        
        balancer = initialize_auto_scaling(
            scaling_configs=scaling_configs,
            load_balancing_strategy="round_robin",
            start_monitoring=False
        )
        
        assert isinstance(balancer, AdaptiveLoadBalancer)
        assert balancer.strategy == "round_robin"
        assert "test_pool" in balancer.scaling_configs
        assert balancer.scaling_configs["test_pool"].min_workers == 2
        assert balancer.scaling_configs["test_pool"].max_workers == 6
        assert not balancer.monitoring
        
        # Cleanup
        balancer.shutdown()


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_high_load_scenario(self):
        """Test system behavior under high load."""
        def slow_worker(task_type, *args, **kwargs):
            time.sleep(0.1)  # Simulate slower work
            return {"result": "completed"}
        
        config = ScalingConfig(
            min_workers=1,
            max_workers=3,
            scale_up_threshold=1,  # Very low threshold
            scale_up_cooldown=0    # No cooldown
        )
        
        pool = WorkerPool(slow_worker, config, initial_workers=1)
        
        try:
            # Submit many tasks quickly
            num_tasks = 10
            for i in range(num_tasks):
                success = pool.submit_task("test_task", i)
                if not success:
                    break  # Queue full
            
            # Should trigger auto-scaling
            initial_workers = pool.target_workers
            pool.auto_scale()
            
            # Should have scaled up
            assert pool.target_workers >= initial_workers
            
        finally:
            pool.shutdown()
    
    def test_error_recovery_scenario(self):
        """Test system behavior during error recovery."""
        error_count = 0
        
        def intermittent_worker(task_type, *args, **kwargs):
            nonlocal error_count
            error_count += 1
            if error_count % 3 == 0:  # Fail every third task
                raise Exception("Intermittent error")
            return {"result": "success"}
        
        config = ScalingConfig(min_workers=1, max_workers=3)
        pool = WorkerPool(intermittent_worker, config, initial_workers=1)
        
        try:
            # Submit tasks that will have some failures
            for i in range(6):
                pool.submit_task("test_task", i)
            
            # Collect results (some will be errors)
            success_count = 0
            error_count_received = 0
            
            for i in range(6):
                result = pool.get_result(timeout=2)
                if result:
                    if result["success"]:
                        success_count += 1
                    else:
                        error_count_received += 1
            
            # Should have both successes and errors
            assert success_count > 0
            assert error_count_received > 0
            
            # Pool should still be functional
            stats = pool.get_statistics()
            assert stats["total_requests"] >= 6
            
        finally:
            pool.shutdown()
    
    def test_resource_exhaustion_scenario(self):
        """Test system behavior during resource exhaustion."""
        def memory_intensive_worker(task_type, *args, **kwargs):
            # Simulate memory-intensive work
            large_data = [0] * 10000  # Small array for testing
            return {"result": "completed", "data_size": len(large_data)}
        
        config = ScalingConfig(
            min_workers=1,
            max_workers=2,
            scale_up_threshold=50.0  # Moderate threshold
        )
        
        pool = WorkerPool(memory_intensive_worker, config, initial_workers=1)
        
        try:
            # Submit multiple memory-intensive tasks
            for i in range(5):
                pool.submit_task("memory_task", i)
            
            # Collect results
            results = []
            for i in range(5):
                result = pool.get_result(timeout=3)
                if result:
                    results.append(result)
            
            # Should have completed most tasks
            assert len(results) >= 3
            
            # Pool should be functional
            stats = pool.get_statistics()
            assert stats["current_workers"] >= 1
            
        finally:
            pool.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])