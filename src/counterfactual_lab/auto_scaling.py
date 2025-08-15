"""Auto-scaling and performance optimization for counterfactual generation."""

import threading
import time
import math
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadMetrics:
    """System load metrics for auto-scaling decisions."""
    timestamp: str
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: Optional[float]
    queue_depth: int
    active_workers: int
    requests_per_second: float
    average_response_time: float
    error_rate: float


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    min_workers: int = 1
    max_workers: int = 8
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 50.0
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 300  # seconds
    queue_threshold: int = 10
    response_time_threshold: float = 30.0  # seconds
    enable_gpu_scaling: bool = True
    enable_process_scaling: bool = False  # Use processes instead of threads


class WorkerPool:
    """Adaptive worker pool with auto-scaling capabilities."""
    
    def __init__(self, 
                 worker_function: Callable,
                 scaling_config: ScalingConfig,
                 initial_workers: int = 2):
        """Initialize worker pool.
        
        Args:
            worker_function: Function to execute in workers
            scaling_config: Auto-scaling configuration
            initial_workers: Initial number of workers
        """
        self.worker_function = worker_function
        self.config = scaling_config
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Worker management
        self.current_workers = 0
        self.target_workers = max(initial_workers, scaling_config.min_workers)
        self.workers: List[threading.Thread] = []
        self.worker_stats: Dict[str, Any] = {}
        
        # Executor for process-based scaling
        if scaling_config.enable_process_scaling:
            self.executor = ProcessPoolExecutor(max_workers=self.target_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.target_workers)
        
        # Metrics tracking
        self.load_metrics: List[LoadMetrics] = []
        self.request_times: List[float] = []
        self.error_count = 0
        self.total_requests = 0
        
        # Scaling state
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min
        self.is_running = False
        
        # Start initial workers
        self._scale_to_target()
        
        logger.info(f"Initialized worker pool with {self.target_workers} workers")
    
    def _scale_to_target(self):
        """Scale workers to target count."""
        current_count = len(self.workers)
        
        if self.target_workers > current_count:
            # Scale up
            for i in range(self.target_workers - current_count):
                self._add_worker()
        elif self.target_workers < current_count:
            # Scale down
            workers_to_remove = current_count - self.target_workers
            for i in range(workers_to_remove):
                self._remove_worker()
        
        # Update executor if needed
        if hasattr(self.executor, '_max_workers'):
            if self.executor._max_workers != self.target_workers:
                self.executor._max_workers = self.target_workers
    
    def _add_worker(self):
        """Add a new worker thread."""
        worker_id = f"worker_{len(self.workers)}"
        worker = threading.Thread(
            target=self._worker_loop,
            args=(worker_id,),
            daemon=True
        )
        worker.start()
        self.workers.append(worker)
        self.worker_stats[worker_id] = {
            "start_time": datetime.now(),
            "tasks_completed": 0,
            "errors": 0
        }
        logger.debug(f"Added worker: {worker_id}")
    
    def _remove_worker(self):
        """Remove a worker thread (graceful shutdown)."""
        if self.workers:
            # Signal worker to stop by putting None in queue
            self.task_queue.put(None)
            worker = self.workers.pop()
            # Note: We don't join here to avoid blocking
            logger.debug(f"Signaled worker removal")
    
    def _worker_loop(self, worker_id: str):
        """Main worker loop."""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                
                if task is None:
                    # Shutdown signal
                    break
                
                start_time = time.time()
                
                try:
                    # Execute task
                    args, kwargs = task.get('args', ()), task.get('kwargs', {})
                    result = self.worker_function(*args, **kwargs)
                    
                    # Record success
                    processing_time = time.time() - start_time
                    self.request_times.append(processing_time)
                    self.worker_stats[worker_id]["tasks_completed"] += 1
                    
                    # Put result in result queue
                    self.result_queue.put({
                        'success': True,
                        'result': result,
                        'processing_time': processing_time,
                        'worker_id': worker_id
                    })
                    
                except Exception as e:
                    # Record error
                    self.error_count += 1
                    self.worker_stats[worker_id]["errors"] += 1
                    
                    self.result_queue.put({
                        'success': False,
                        'error': str(e),
                        'processing_time': time.time() - start_time,
                        'worker_id': worker_id
                    })
                
                finally:
                    self.task_queue.task_done()
                    self.total_requests += 1
                    
            except queue.Empty:
                # Timeout waiting for task - check if we should continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                break
        
        logger.debug(f"Worker {worker_id} shutting down")
    
    def submit_task(self, *args, **kwargs) -> bool:
        """Submit a task to the worker pool.
        
        Returns:
            True if task was submitted successfully
        """
        try:
            task = {'args': args, 'kwargs': kwargs}
            self.task_queue.put(task, timeout=1)
            return True
        except queue.Full:
            logger.warning("Task queue full, rejecting task")
            return False
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get a result from the worker pool.
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            Result dictionary or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_load_metrics(self) -> LoadMetrics:
        """Get current load metrics."""
        try:
            import psutil
            cpu_util = psutil.cpu_percent(interval=0.1)
            memory_util = psutil.virtual_memory().percent
        except ImportError:
            cpu_util = 0.0
            memory_util = 0.0
        
        # GPU utilization if available
        gpu_util = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        except (ImportError, RuntimeError):
            pass
        
        # Calculate requests per second
        recent_requests = len([
            t for t in self.request_times[-100:]
            if time.time() - t < 60
        ])
        rps = recent_requests / 60.0
        
        # Calculate average response time
        recent_times = self.request_times[-50:] if self.request_times else [0]
        avg_response_time = sum(recent_times) / len(recent_times)
        
        # Calculate error rate
        recent_total = max(self.total_requests, 1)
        error_rate = self.error_count / recent_total * 100
        
        return LoadMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            gpu_utilization=gpu_util,
            queue_depth=self.task_queue.qsize(),
            active_workers=len(self.workers),
            requests_per_second=rps,
            average_response_time=avg_response_time,
            error_rate=error_rate
        )
    
    def should_scale_up(self, metrics: LoadMetrics) -> bool:
        """Determine if we should scale up workers."""
        # Check cooldown
        if (datetime.now() - self.last_scale_up).total_seconds() < self.config.scale_up_cooldown:
            return False
        
        # Check if at max workers
        if self.target_workers >= self.config.max_workers:
            return False
        
        # Check scaling triggers
        triggers = [
            metrics.cpu_utilization > self.config.scale_up_threshold,
            metrics.memory_utilization > self.config.scale_up_threshold,
            metrics.queue_depth > self.config.queue_threshold,
            metrics.average_response_time > self.config.response_time_threshold,
            metrics.error_rate > 5.0  # Scale up if error rate is high
        ]
        
        # GPU scaling
        if self.config.enable_gpu_scaling and metrics.gpu_utilization:
            triggers.append(metrics.gpu_utilization > self.config.scale_up_threshold)
        
        return any(triggers)
    
    def should_scale_down(self, metrics: LoadMetrics) -> bool:
        """Determine if we should scale down workers."""
        # Check cooldown
        if (datetime.now() - self.last_scale_down).total_seconds() < self.config.scale_down_cooldown:
            return False
        
        # Check if at min workers
        if self.target_workers <= self.config.min_workers:
            return False
        
        # Check scaling triggers (all must be true for scale down)
        conditions = [
            metrics.cpu_utilization < self.config.scale_down_threshold,
            metrics.memory_utilization < self.config.scale_down_threshold,
            metrics.queue_depth < 2,
            metrics.average_response_time < self.config.response_time_threshold / 2,
            metrics.error_rate < 1.0
        ]
        
        # GPU scaling
        if self.config.enable_gpu_scaling and metrics.gpu_utilization:
            conditions.append(metrics.gpu_utilization < self.config.scale_down_threshold)
        
        return all(conditions)
    
    def auto_scale(self) -> bool:
        """Perform auto-scaling based on current metrics.
        
        Returns:
            True if scaling action was taken
        """
        metrics = self.get_load_metrics()
        self.load_metrics.append(metrics)
        
        # Keep only recent metrics
        if len(self.load_metrics) > 100:
            self.load_metrics = self.load_metrics[-100:]
        
        action_taken = False
        
        if self.should_scale_up(metrics):
            old_target = self.target_workers
            self.target_workers = min(
                self.target_workers + 1,
                self.config.max_workers
            )
            
            if self.target_workers > old_target:
                self._scale_to_target()
                self.last_scale_up = datetime.now()
                action_taken = True
                logger.info(f"Scaled up: {old_target} -> {self.target_workers} workers")
        
        elif self.should_scale_down(metrics):
            old_target = self.target_workers
            self.target_workers = max(
                self.target_workers - 1,
                self.config.min_workers
            )
            
            if self.target_workers < old_target:
                self._scale_to_target()
                self.last_scale_down = datetime.now()
                action_taken = True
                logger.info(f"Scaled down: {old_target} -> {self.target_workers} workers")
        
        return action_taken
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        recent_metrics = self.load_metrics[-10:] if self.load_metrics else []
        
        return {
            "current_workers": len(self.workers),
            "target_workers": self.target_workers,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1) * 100,
            "queue_depth": self.task_queue.qsize(),
            "worker_stats": self.worker_stats,
            "recent_load": [metrics.__dict__ for metrics in recent_metrics],
            "average_response_time": sum(self.request_times[-50:]) / len(self.request_times[-50:]) if self.request_times else 0,
            "scaling_config": self.config.__dict__
        }
    
    def shutdown(self):
        """Shutdown the worker pool gracefully."""
        logger.info("Shutting down worker pool...")
        
        # Signal all workers to stop
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Worker pool shutdown complete")


class AdaptiveLoadBalancer:
    """Adaptive load balancer for distributing generation tasks."""
    
    def __init__(self, 
                 scaling_configs: Dict[str, ScalingConfig],
                 load_balancing_strategy: str = "least_loaded"):
        """Initialize adaptive load balancer.
        
        Args:
            scaling_configs: Scaling configs for different task types
            load_balancing_strategy: Strategy for load balancing
        """
        self.scaling_configs = scaling_configs
        self.strategy = load_balancing_strategy
        self.worker_pools: Dict[str, WorkerPool] = {}
        self.task_routing: Dict[str, str] = {}
        self.global_metrics: List[LoadMetrics] = []
        
        # Auto-scaling monitor
        self.monitor_thread = None
        self.monitoring = False
        
        logger.info(f"Initialized adaptive load balancer with {load_balancing_strategy} strategy")
    
    def register_worker_pool(self, 
                           pool_name: str,
                           worker_function: Callable,
                           task_types: List[str]):
        """Register a worker pool for specific task types.
        
        Args:
            pool_name: Name of the worker pool
            worker_function: Function to execute in workers
            task_types: List of task types this pool handles
        """
        if pool_name not in self.scaling_configs:
            self.scaling_configs[pool_name] = ScalingConfig()
        
        self.worker_pools[pool_name] = WorkerPool(
            worker_function=worker_function,
            scaling_config=self.scaling_configs[pool_name]
        )
        
        # Map task types to this pool
        for task_type in task_types:
            self.task_routing[task_type] = pool_name
        
        logger.info(f"Registered worker pool '{pool_name}' for tasks: {task_types}")
    
    def submit_task(self, 
                   task_type: str,
                   *args, **kwargs) -> Optional[str]:
        """Submit a task to appropriate worker pool.
        
        Args:
            task_type: Type of task to execute
            *args, **kwargs: Task arguments
            
        Returns:
            Pool name where task was submitted, or None if failed
        """
        # Route task to appropriate pool
        if task_type not in self.task_routing:
            logger.error(f"No worker pool registered for task type: {task_type}")
            return None
        
        pool_name = self.task_routing[task_type]
        
        # Apply load balancing if multiple pools available
        if self.strategy == "least_loaded":
            pool_name = self._select_least_loaded_pool(task_type)
        elif self.strategy == "round_robin":
            pool_name = self._select_round_robin_pool(task_type)
        
        pool = self.worker_pools[pool_name]
        success = pool.submit_task(*args, **kwargs)
        
        return pool_name if success else None
    
    def _select_least_loaded_pool(self, task_type: str) -> str:
        """Select least loaded pool for task type."""
        # Get all pools that can handle this task type
        candidate_pools = [
            name for name, pool_name in self.task_routing.items()
            if name == task_type
        ]
        
        if len(candidate_pools) <= 1:
            return self.task_routing[task_type]
        
        # Find pool with lowest load
        min_load = float('inf')
        best_pool = self.task_routing[task_type]
        
        for pool_name in set(self.task_routing[t] for t in candidate_pools):
            pool = self.worker_pools[pool_name]
            metrics = pool.get_load_metrics()
            
            # Calculate load score
            load_score = (
                metrics.cpu_utilization * 0.4 +
                metrics.memory_utilization * 0.3 +
                metrics.queue_depth * 10 +
                metrics.average_response_time * 0.3
            )
            
            if load_score < min_load:
                min_load = load_score
                best_pool = pool_name
        
        return best_pool
    
    def _select_round_robin_pool(self, task_type: str) -> str:
        """Select pool using round-robin strategy."""
        # Simple implementation - would need to track state for true round-robin
        return self.task_routing[task_type]
    
    def start_monitoring(self, interval: int = 30):
        """Start auto-scaling monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started auto-scaling monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped auto-scaling monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Auto-scaling monitoring loop."""
        while self.monitoring:
            try:
                # Collect global metrics
                global_cpu = 0.0
                global_memory = 0.0
                global_queue_depth = 0
                total_workers = 0
                
                # Auto-scale each pool
                for pool_name, pool in self.worker_pools.items():
                    try:
                        action_taken = pool.auto_scale()
                        if action_taken:
                            logger.info(f"Auto-scaling action taken for pool: {pool_name}")
                        
                        # Collect metrics for global view
                        metrics = pool.get_load_metrics()
                        global_cpu += metrics.cpu_utilization
                        global_memory += metrics.memory_utilization
                        global_queue_depth += metrics.queue_depth
                        total_workers += metrics.active_workers
                        
                    except Exception as e:
                        logger.error(f"Error during auto-scaling for pool {pool_name}: {e}")
                
                # Store global metrics
                if self.worker_pools:
                    pool_count = len(self.worker_pools)
                    global_metrics = LoadMetrics(
                        timestamp=datetime.now().isoformat(),
                        cpu_utilization=global_cpu / pool_count,
                        memory_utilization=global_memory / pool_count,
                        gpu_utilization=None,  # Would need aggregation logic
                        queue_depth=global_queue_depth,
                        active_workers=total_workers,
                        requests_per_second=0.0,  # Would need aggregation
                        average_response_time=0.0,  # Would need aggregation
                        error_rate=0.0  # Would need aggregation
                    )
                    self.global_metrics.append(global_metrics)
                    
                    # Keep only recent metrics
                    if len(self.global_metrics) > 100:
                        self.global_metrics = self.global_metrics[-100:]
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global load balancer statistics."""
        pool_stats = {}
        for pool_name, pool in self.worker_pools.items():
            pool_stats[pool_name] = pool.get_statistics()
        
        return {
            "total_pools": len(self.worker_pools),
            "total_workers": sum(stats["current_workers"] for stats in pool_stats.values()),
            "total_requests": sum(stats["total_requests"] for stats in pool_stats.values()),
            "global_error_rate": sum(stats["error_count"] for stats in pool_stats.values()) / 
                                max(sum(stats["total_requests"] for stats in pool_stats.values()), 1) * 100,
            "pool_statistics": pool_stats,
            "task_routing": self.task_routing,
            "global_metrics": [metrics.__dict__ for metrics in self.global_metrics[-10:]],
            "monitoring_active": self.monitoring
        }
    
    def shutdown(self):
        """Shutdown all worker pools and monitoring."""
        logger.info("Shutting down adaptive load balancer...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Shutdown all pools
        for pool_name, pool in self.worker_pools.items():
            try:
                pool.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down pool {pool_name}: {e}")
        
        logger.info("Adaptive load balancer shutdown complete")


# Global load balancer instance
_global_load_balancer = None

def get_global_load_balancer() -> AdaptiveLoadBalancer:
    """Get or create global load balancer."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = AdaptiveLoadBalancer({})
    return _global_load_balancer

def initialize_auto_scaling(scaling_configs: Optional[Dict[str, ScalingConfig]] = None,
                          load_balancing_strategy: str = "least_loaded",
                          start_monitoring: bool = True) -> AdaptiveLoadBalancer:
    """Initialize global auto-scaling system.
    
    Args:
        scaling_configs: Scaling configurations for different pools
        load_balancing_strategy: Load balancing strategy
        start_monitoring: Whether to start monitoring immediately
        
    Returns:
        Global load balancer instance
    """
    global _global_load_balancer
    
    configs = scaling_configs or {
        "generation": ScalingConfig(min_workers=1, max_workers=8),
        "evaluation": ScalingConfig(min_workers=1, max_workers=4)
    }
    
    _global_load_balancer = AdaptiveLoadBalancer(
        scaling_configs=configs,
        load_balancing_strategy=load_balancing_strategy
    )
    
    if start_monitoring:
        _global_load_balancer.start_monitoring()
    
    logger.info("Auto-scaling system initialized")
    return _global_load_balancer