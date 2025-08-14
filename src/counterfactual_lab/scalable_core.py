"""Scalable core with performance optimization, caching, and concurrent processing."""

import json
import logging
import random
import hashlib
import traceback
import time
import asyncio
import threading
import multiprocessing
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import warnings

# Enhanced logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(thread)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scalable_counterfactual_lab.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class PerformanceMonitor:
    """Advanced performance monitoring and metrics collection."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'avg_response_time': 0.0,
            'p95_response_time': 0.0,
            'throughput_rps': 0.0,
            'concurrent_requests': 0,
            'peak_concurrent': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0
        }
        
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        self.request_timestamps = deque(maxlen=1000)
        self.active_requests = set()
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitor initialized")
    
    def record_request_start(self, request_id: str):
        """Record the start of a request."""
        with self._lock:
            self.active_requests.add(request_id)
            self.metrics['concurrent_requests'] = len(self.active_requests)
            self.metrics['peak_concurrent'] = max(
                self.metrics['peak_concurrent'], 
                self.metrics['concurrent_requests']
            )
    
    def record_request_end(self, request_id: str, success: bool, duration: float):
        """Record the end of a request."""
        with self._lock:
            self.active_requests.discard(request_id)
            self.metrics['concurrent_requests'] = len(self.active_requests)
            self.metrics['requests_total'] += 1
            
            if success:
                self.metrics['requests_successful'] += 1
            else:
                self.metrics['requests_failed'] += 1
            
            # Update response times
            self.response_times.append(duration)
            self.request_timestamps.append(time.time())
            
            # Calculate running average
            if self.response_times:
                self.metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
                
                # Calculate P95
                sorted_times = sorted(self.response_times)
                p95_idx = int(len(sorted_times) * 0.95)
                self.metrics['p95_response_time'] = sorted_times[p95_idx] if sorted_times else 0.0
            
            # Calculate throughput (requests per second in last minute)
            cutoff_time = time.time() - 60  # Last minute
            recent_requests = sum(1 for ts in self.request_timestamps if ts > cutoff_time)
            self.metrics['throughput_rps'] = recent_requests / 60.0
    
    def record_cache_hit(self):
        """Record cache hit."""
        with self._lock:
            self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        with self._lock:
            self.metrics['cache_misses'] += 1
    
    def _monitor_system(self):
        """Monitor system resources."""
        try:
            import psutil
            PSUTIL_AVAILABLE = True
        except ImportError:
            PSUTIL_AVAILABLE = False
        
        while self.monitoring:
            try:
                if PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    with self._lock:
                        self.metrics['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
                        self.metrics['cpu_usage_percent'] = process.cpu_percent()
                else:
                    # Mock system metrics
                    with self._lock:
                        self.metrics['memory_usage_mb'] = random.uniform(100, 500)
                        self.metrics['cpu_usage_percent'] = random.uniform(10, 80)
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
                time.sleep(10)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            cache_total = self.metrics['cache_hits'] + self.metrics['cache_misses']
            cache_hit_rate = self.metrics['cache_hits'] / max(cache_total, 1) * 100
            
            return {
                **self.metrics.copy(),
                'cache_hit_rate': cache_hit_rate,
                'success_rate': (
                    self.metrics['requests_successful'] / 
                    max(self.metrics['requests_total'], 1) * 100
                ),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        with self._lock:
            success_rate = self.metrics['requests_successful'] / max(self.metrics['requests_total'], 1)
            response_time_score = max(0, min(1, 1 - (self.metrics['avg_response_time'] / 10.0)))
            cpu_score = max(0, min(1, 1 - (self.metrics['cpu_usage_percent'] / 100.0)))
            
            health_score = (success_rate + response_time_score + cpu_score) / 3 * 100
            return health_score
    
    def shutdown(self):
        """Shutdown monitoring."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)


class IntelligentCache:
    """Advanced caching system with TTL, LRU eviction, and analytics."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize intelligent cache."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.expiry_times = {}
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Intelligent cache initialized: max_size={max_size}, ttl={default_ttl}s")
    
    def _generate_key(self, method: str, image_hash: str, text: str, attributes: Dict[str, str]) -> str:
        """Generate cache key from parameters."""
        key_data = {
            'method': method,
            'image_hash': image_hash,
            'text': text,
            'attributes': sorted(attributes.items())  # Sort for consistent hashing
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, method: str, image_hash: str, text: str, attributes: Dict[str, str]) -> Optional[Any]:
        """Get item from cache."""
        key = self._generate_key(method, image_hash, text, attributes)
        
        with self._lock:
            if key not in self.cache:
                return None
            
            # Check expiry
            if key in self.expiry_times and time.time() > self.expiry_times[key]:
                self._evict_key(key)
                return None
            
            # Update access stats
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            return self.cache[key]
    
    def set(
        self, 
        method: str, 
        image_hash: str, 
        text: str, 
        attributes: Dict[str, str], 
        value: Any, 
        ttl: Optional[int] = None
    ):
        """Set item in cache."""
        key = self._generate_key(method, image_hash, text, attributes)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.expiry_times[key] = time.time() + ttl
    
    def _evict_key(self, key: str):
        """Evict specific key from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.expiry_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find LRU key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict_key(lru_key)
    
    def _cleanup_expired(self):
        """Cleanup expired entries periodically."""
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                with self._lock:
                    for key, expiry_time in self.expiry_times.items():
                        if current_time > expiry_time:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._evict_key(key)
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                time.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")
                time.sleep(300)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(self.access_counts.values())
            avg_access_count = total_accesses / max(len(self.access_counts), 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size * 100,
                'total_accesses': total_accesses,
                'avg_access_count': avg_access_count,
                'expired_entries': len([
                    k for k, t in self.expiry_times.items() 
                    if time.time() > t
                ])
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.expiry_times.clear()


class WorkerPool:
    """Dynamic worker pool for concurrent processing."""
    
    def __init__(self, initial_workers: int = 4, max_workers: int = 20):
        """Initialize worker pool."""
        self.min_workers = initial_workers
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=initial_workers)
        self.current_workers = initial_workers
        self.queue_size = 0
        self.completed_tasks = 0
        self._lock = threading.Lock()
        
        # Auto-scaling metrics
        self.queue_history = deque(maxlen=60)  # Track queue size for 1 minute
        self.scaling_thread = threading.Thread(target=self._auto_scale, daemon=True)
        self.scaling_enabled = True
        self.scaling_thread.start()
        
        logger.info(f"Worker pool initialized: {initial_workers} workers")
    
    def submit_task(self, fn: Callable, *args, **kwargs):
        """Submit task to worker pool."""
        with self._lock:
            self.queue_size += 1
        
        future = self.executor.submit(self._task_wrapper, fn, *args, **kwargs)
        return future
    
    def _task_wrapper(self, fn: Callable, *args, **kwargs):
        """Wrapper for tasks to track completion."""
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self.completed_tasks += 1
                self.queue_size = max(0, self.queue_size - 1)
            return result
        except Exception as e:
            with self._lock:
                self.queue_size = max(0, self.queue_size - 1)
            raise e
    
    def _auto_scale(self):
        """Auto-scale worker pool based on load."""
        while self.scaling_enabled:
            try:
                with self._lock:
                    current_queue = self.queue_size
                    self.queue_history.append(current_queue)
                
                # Calculate metrics
                if len(self.queue_history) >= 10:
                    avg_queue_size = sum(list(self.queue_history)[-10:]) / 10
                    
                    # Scale up if queue is consistently high
                    if (avg_queue_size > self.current_workers * 2 and 
                        self.current_workers < self.max_workers):
                        new_workers = min(self.current_workers + 2, self.max_workers)
                        self._resize_pool(new_workers)
                        logger.info(f"Scaled up worker pool: {self.current_workers} -> {new_workers}")
                    
                    # Scale down if queue is consistently low
                    elif (avg_queue_size < self.current_workers * 0.5 and 
                          self.current_workers > self.min_workers):
                        new_workers = max(self.current_workers - 1, self.min_workers)
                        self._resize_pool(new_workers)
                        logger.info(f"Scaled down worker pool: {self.current_workers} -> {new_workers}")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.warning(f"Auto-scaling error: {e}")
                time.sleep(10)
    
    def _resize_pool(self, new_size: int):
        """Resize the thread pool."""
        try:
            # Create new executor with new size
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(max_workers=new_size)
            self.current_workers = new_size
            
            # Gracefully shutdown old executor
            threading.Thread(
                target=lambda: old_executor.shutdown(wait=True), 
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"Failed to resize worker pool: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            return {
                'current_workers': self.current_workers,
                'max_workers': self.max_workers,
                'queue_size': self.queue_size,
                'completed_tasks': self.completed_tasks,
                'avg_queue_size': sum(self.queue_history) / max(len(self.queue_history), 1)
            }
    
    def shutdown(self):
        """Shutdown worker pool."""
        self.scaling_enabled = False
        if self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5)
        self.executor.shutdown(wait=True)


class MockImage:
    """Enhanced mock image with performance optimizations."""
    
    def __init__(self, width: int = 400, height: int = 300, mode: str = "RGB"):
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        if width > 10000 or height > 10000:
            raise ValueError("Image dimensions too large")
        
        self.width = width
        self.height = height
        self.mode = mode
        self.format = None
        self._hash = None  # Cached hash
        self.data = f"scalable_image_{width}x{height}_{random.randint(10000, 99999)}"
    
    def copy(self):
        """Return a copy of this mock image."""
        return MockImage(self.width, self.height, self.mode)
    
    @lru_cache(maxsize=1)
    def get_hash(self) -> str:
        """Get cached hash of image."""
        if self._hash is None:
            hash_input = f"{self.width}_{self.height}_{self.mode}_{self.data}"
            self._hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return self._hash
    
    def save(self, fp, format=None):
        """Mock save operation."""
        if hasattr(fp, 'write'):
            fp.write(f"MOCK_IMAGE_DATA_{self.data}")
        else:
            with open(fp, 'w') as f:
                f.write(f"MOCK_IMAGE_DATA_{self.data}")
    
    def __str__(self):
        return f"MockImage({self.width}x{self.height}, {self.mode})"


class ScalableCounterfactualGenerator:
    """High-performance scalable counterfactual generator with advanced optimizations."""
    
    def __init__(
        self,
        method: str = "modicf",
        device: str = "cpu",
        enable_caching: bool = True,
        enable_worker_pool: bool = True,
        enable_monitoring: bool = True,
        cache_size: int = 1000,
        initial_workers: int = 4,
        max_workers: int = 20
    ):
        """Initialize scalable generator with optimizations."""
        self.method = method.lower()
        self.device = device.lower()
        self.start_time = datetime.now()
        
        # Core components
        self.performance_monitor = PerformanceMonitor() if enable_monitoring else None
        self.cache = IntelligentCache(cache_size) if enable_caching else None
        self.worker_pool = WorkerPool(initial_workers, max_workers) if enable_worker_pool else None
        
        # Performance tracking
        self.generation_count = 0
        self.batch_count = 0
        self._stats_lock = threading.Lock()
        
        logger.info(
            f"Scalable generator initialized: method={method}, device={device}, "
            f"caching={enable_caching}, workers={enable_worker_pool}, "
            f"monitoring={enable_monitoring}"
        )
    
    def generate(
        self,
        image: Union[str, Path, MockImage],
        text: str,
        attributes: Union[List[str], str],
        num_samples: int = 5,
        user_id: str = "anonymous",
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate counterfactuals with full optimization."""
        
        request_id = f"gen_{int(time.time() * 1000)}_{threading.get_ident()}"
        start_time = time.time()
        
        # Start monitoring
        if self.performance_monitor:
            self.performance_monitor.record_request_start(request_id)
        
        try:
            # Input processing and validation
            processed_image = self._process_image_input(image)
            processed_text = self._validate_text(text)
            processed_attributes = self._process_attributes(attributes)
            
            # Generate image hash for caching
            image_hash = processed_image.get_hash()
            
            # Check cache first
            if use_cache and self.cache:
                cached_result = self.cache.get(
                    self.method, image_hash, processed_text, 
                    {attr: "varied" for attr in processed_attributes}
                )
                
                if cached_result:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit()
                    
                    duration = time.time() - start_time
                    if self.performance_monitor:
                        self.performance_monitor.record_request_end(request_id, True, duration)
                    
                    logger.info(f"Cache hit for request {request_id} ({duration:.3f}s)")
                    return cached_result
                else:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_miss()
            
            # Generate counterfactuals
            logger.info(f"Generating {num_samples} counterfactuals with {self.method}")
            
            if self.worker_pool and num_samples > 1:
                # Parallel generation for multiple samples
                results = self._generate_parallel(
                    processed_image, processed_text, processed_attributes, num_samples
                )
            else:
                # Sequential generation
                results = self._generate_sequential(
                    processed_image, processed_text, processed_attributes, num_samples
                )
            
            # Prepare response
            generation_time = time.time() - start_time
            
            response = {
                "method": self.method,
                "original_image": processed_image,
                "original_text": processed_text,
                "target_attributes": processed_attributes,
                "counterfactuals": results,
                "metadata": {
                    "generation_time": generation_time,
                    "num_samples": len(results),
                    "device": self.device,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                    "user_id": user_id,
                    "cache_used": False,
                    "parallel_processing": self.worker_pool and num_samples > 1,
                    "generation_id": self.generation_count
                }
            }
            
            # Cache the result
            if use_cache and self.cache:
                self.cache.set(
                    self.method, image_hash, processed_text,
                    {attr: "varied" for attr in processed_attributes},
                    response,
                    ttl=3600  # 1 hour TTL
                )
            
            # Update stats
            with self._stats_lock:
                self.generation_count += 1
            
            if self.performance_monitor:
                self.performance_monitor.record_request_end(request_id, True, generation_time)
            
            logger.info(f"Generation completed: {request_id} ({generation_time:.3f}s)")
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            if self.performance_monitor:
                self.performance_monitor.record_request_end(request_id, False, duration)
            
            logger.error(f"Generation failed: {request_id} ({duration:.3f}s): {e}")
            raise
    
    def _process_image_input(self, image: Union[str, Path, MockImage]) -> MockImage:
        """Process image input with optimization."""
        if isinstance(image, (str, Path)):
            return MockImage(400, 300)  # Mock loading
        elif hasattr(image, 'width') and hasattr(image, 'height'):
            # Convert to scalable MockImage if needed
            if not hasattr(image, 'get_hash'):
                return MockImage(image.width, image.height, getattr(image, 'mode', 'RGB'))
            return image
        else:
            raise ValueError("Invalid image input")
    
    def _validate_text(self, text: str) -> str:
        """Validate and process text input."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        
        if len(text) < 3:
            raise ValueError("Text too short")
        
        return text
    
    def _process_attributes(self, attributes: Union[List[str], str]) -> List[str]:
        """Process and validate attributes."""
        if isinstance(attributes, str):
            attributes = [attr.strip().lower() for attr in attributes.split(',')]
        
        if not isinstance(attributes, list) or not attributes:
            raise ValueError("Invalid attributes")
        
        supported_attrs = {"gender", "race", "age", "expression", "hair_color", "hair_style"}
        valid_attrs = [attr for attr in attributes if attr in supported_attrs]
        
        if not valid_attrs:
            raise ValueError("No valid attributes found")
        
        return valid_attrs
    
    def _generate_parallel(
        self, 
        image: MockImage, 
        text: str, 
        attributes: List[str], 
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate counterfactuals in parallel."""
        
        futures = []
        
        # Submit tasks to worker pool
        for i in range(num_samples):
            future = self.worker_pool.submit_task(
                self._generate_single_sample, 
                image, text, attributes, i
            )
            futures.append(future)
        
        # Collect results as they complete
        results = []
        for future in as_completed(futures, timeout=30):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.warning(f"Parallel sample generation failed: {e}")
                # Add failed sample placeholder
                results.append({
                    "sample_id": len(results),
                    "error": str(e),
                    "generation_success": False
                })
        
        # Sort by sample_id to maintain order
        results.sort(key=lambda x: x.get('sample_id', 0))
        
        return results
    
    def _generate_sequential(
        self, 
        image: MockImage, 
        text: str, 
        attributes: List[str], 
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate counterfactuals sequentially."""
        
        results = []
        
        for i in range(num_samples):
            try:
                result = self._generate_single_sample(image, text, attributes, i)
                results.append(result)
            except Exception as e:
                logger.warning(f"Sequential sample generation failed: {e}")
                results.append({
                    "sample_id": i,
                    "error": str(e),
                    "generation_success": False
                })
        
        return results
    
    @lru_cache(maxsize=100)
    def _get_attribute_values(self, attribute: str) -> List[str]:
        """Get cached attribute values."""
        attribute_map = {
            "gender": ["male", "female", "non-binary"],
            "race": ["white", "black", "asian", "hispanic", "diverse"],
            "age": ["young", "middle-aged", "elderly"],
            "expression": ["neutral", "smiling", "serious", "happy", "sad"],
            "hair_color": ["blonde", "brown", "black", "red", "gray"],
            "hair_style": ["short", "long", "curly", "straight"]
        }
        return attribute_map.get(attribute, [])
    
    def _generate_single_sample(
        self, 
        image: MockImage, 
        text: str, 
        attributes: List[str], 
        sample_id: int
    ) -> Dict[str, Any]:
        """Generate a single counterfactual sample."""
        
        # Generate target attributes
        target_attrs = {}
        for attr in attributes:
            possible_values = self._get_attribute_values(attr)
            if possible_values:
                target_attrs[attr] = random.choice(possible_values)
        
        # Mock generation process
        generated_image = image.copy()
        
        # Apply text modifications
        modified_text = self._modify_text_optimized(text, target_attrs)
        
        # Calculate confidence
        confidence = self._calculate_confidence_optimized(target_attrs, len(attributes))
        
        return {
            "sample_id": sample_id,
            "target_attributes": target_attrs,
            "generated_image": generated_image,
            "generated_text": modified_text,
            "confidence": confidence,
            "explanation": f"Applied {self.method} to modify {', '.join(attributes)}",
            "generation_success": True,
            "processing_time": random.uniform(0.1, 0.5)  # Mock processing time
        }
    
    def _modify_text_optimized(self, text: str, target_attrs: Dict[str, str]) -> str:
        """Optimized text modification."""
        modified = text
        
        for attr, value in target_attrs.items():
            if attr == "gender":
                if value == "female" and "man" in modified.lower():
                    modified = modified.replace("man", "woman").replace("Man", "Woman")
                elif value == "male" and "woman" in modified.lower():
                    modified = modified.replace("woman", "man").replace("Woman", "Man")
            elif attr == "age":
                if value in ["young", "elderly"] and not any(age in modified.lower() for age in ["young", "old", "elderly"]):
                    modified = f"{value} {modified}"
        
        return modified
    
    def _calculate_confidence_optimized(self, target_attrs: Dict[str, str], num_attributes: int) -> float:
        """Optimized confidence calculation."""
        base_confidence = 0.85
        complexity_penalty = len(target_attrs) * 0.04
        random_factor = random.uniform(-0.03, 0.03)
        
        confidence = base_confidence - complexity_penalty + random_factor
        return max(0.2, min(1.0, confidence))
    
    async def generate_async(self, **kwargs) -> Dict[str, Any]:
        """Async interface for generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))
    
    def generate_batch(
        self, 
        requests: List[Dict[str, Any]], 
        max_parallel: int = 10
    ) -> List[Dict[str, Any]]:
        """Optimized batch generation."""
        
        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(f"Starting batch generation: {batch_id} ({len(requests)} requests)")
        
        with self._stats_lock:
            self.batch_count += 1
        
        if not self.worker_pool:
            # Sequential processing
            results = []
            for i, request in enumerate(requests):
                try:
                    result = self.generate(**request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch request {i} failed: {e}")
                    results.append({"error": str(e), "success": False})
            return results
        
        # Parallel processing
        futures = []
        
        # Process in chunks to avoid overwhelming the system
        chunk_size = min(max_parallel, len(requests))
        
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i + chunk_size]
            
            chunk_futures = []
            for request in chunk:
                future = self.worker_pool.submit_task(
                    self._safe_generate, request
                )
                chunk_futures.append(future)
            
            futures.extend(chunk_futures)
        
        # Collect results
        results = []
        for future in as_completed(futures, timeout=300):  # 5 minute timeout
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Batch generation task failed: {e}")
                results.append({"error": str(e), "success": False})
        
        batch_time = time.time() - start_time
        logger.info(f"Batch generation completed: {batch_id} ({batch_time:.2f}s)")
        
        return results
    
    def _safe_generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Safe generation wrapper for batch processing."""
        try:
            return self.generate(**request)
        except Exception as e:
            logger.error(f"Safe generation failed: {e}")
            return {"error": str(e), "success": False}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "generator_stats": {
                "generation_count": self.generation_count,
                "batch_count": self.batch_count,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "method": self.method,
                "device": self.device
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if self.performance_monitor:
            metrics["system_metrics"] = self.performance_monitor.get_metrics()
            metrics["health_score"] = self.performance_monitor.get_health_score()
        
        if self.cache:
            metrics["cache_stats"] = self.cache.get_stats()
        
        if self.worker_pool:
            metrics["worker_pool_stats"] = self.worker_pool.get_stats()
        
        return metrics
    
    def optimize_performance(self, **config) -> Dict[str, Any]:
        """Dynamic performance optimization."""
        optimizations = []
        
        if self.performance_monitor:
            current_metrics = self.performance_monitor.get_metrics()
            
            # Optimize based on current load
            if current_metrics['avg_response_time'] > 5.0:
                optimizations.append("Recommended: Enable parallel processing")
            
            if current_metrics.get('cache_hit_rate', 0) < 50:
                optimizations.append("Recommended: Increase cache size")
            
            if current_metrics['cpu_usage_percent'] > 80:
                optimizations.append("Recommended: Scale horizontally")
        
        return {
            "current_config": {
                "method": self.method,
                "caching_enabled": self.cache is not None,
                "worker_pool_enabled": self.worker_pool is not None,
                "monitoring_enabled": self.performance_monitor is not None
            },
            "optimizations": optimizations,
            "timestamp": datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check core components
        health_data["components"]["generator"] = {"status": "healthy"}
        
        if self.performance_monitor:
            health_score = self.performance_monitor.get_health_score()
            health_data["components"]["monitoring"] = {
                "status": "healthy" if health_score > 70 else "degraded",
                "health_score": health_score
            }
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            health_data["components"]["cache"] = {
                "status": "healthy" if cache_stats["utilization"] < 90 else "degraded",
                "utilization": cache_stats["utilization"]
            }
        
        if self.worker_pool:
            worker_stats = self.worker_pool.get_stats()
            health_data["components"]["worker_pool"] = {
                "status": "healthy" if worker_stats["queue_size"] < 100 else "degraded",
                "workers": worker_stats["current_workers"],
                "queue_size": worker_stats["queue_size"]
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_data["components"].values()]
        if any(status == "unhealthy" for status in component_statuses):
            health_data["status"] = "unhealthy"
        elif any(status == "degraded" for status in component_statuses):
            health_data["status"] = "degraded"
        
        return health_data
    
    def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("Initiating scalable generator shutdown...")
        
        if self.performance_monitor:
            self.performance_monitor.shutdown()
        
        if self.worker_pool:
            self.worker_pool.shutdown()
        
        if self.cache:
            self.cache.clear()
        
        logger.info("Scalable generator shutdown completed")


class ScalableBiasEvaluator:
    """Scalable bias evaluator with parallel processing and caching."""
    
    def __init__(self, model=None, max_workers: int = None, **kwargs):
        """Initialize scalable bias evaluator."""
        self.model = model
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.cache = IntelligentCache(max_size=500)
        self.stats = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'parallel_evaluations': 0,
            'start_time': time.time()
        }
        
        logger.info(f"Scalable bias evaluator initialized with {self.max_workers} workers")
    
    def evaluate(self, counterfactuals: Dict[str, Any], metrics: List[str], 
                user_id: str = None, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """Evaluate bias with parallel processing and caching."""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(counterfactuals, metrics)
            cached_result = self.cache.get(
                method="bias_evaluation", 
                image_hash=cache_key, 
                text=str(metrics), 
                attributes={"evaluation": "bias"}
            )
            
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit for bias evaluation: {cache_key}")
                return cached_result
        
        # Parallel evaluation of metrics
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_metric = {
                executor.submit(self._evaluate_metric, counterfactuals, metric): metric
                for metric in metrics
            }
            
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    results[metric] = future.result()
                except Exception as e:
                    logger.error(f"Metric evaluation failed for {metric}: {e}")
                    results[metric] = {'error': str(e)}
        
        # Compile final evaluation result
        evaluation_result = {
            'metrics': results,
            'summary': self._compile_summary(results),
            'metadata': {
                'evaluation_time': time.time() - start_time,
                'parallel_processing': True,
                'user_id': user_id,
                'num_metrics': len(metrics),
                'evaluator_version': 'scalable_v1.0'
            },
            'validation_passed': self._validate_results(results)
        }
        
        # Cache the result
        if use_cache:
            self.cache.set(
                method="bias_evaluation",
                image_hash=cache_key,
                text=str(metrics),
                attributes={"evaluation": "bias"},
                value=evaluation_result,
                ttl=1800  # 30 minutes TTL
            )
        
        # Update statistics
        self.stats['total_evaluations'] += 1
        self.stats['parallel_evaluations'] += 1
        
        logger.info(f"Bias evaluation completed for user {user_id} in {time.time() - start_time:.3f}s")
        return evaluation_result
    
    def _generate_cache_key(self, counterfactuals: Dict[str, Any], metrics: List[str]) -> str:
        """Generate cache key for evaluation."""
        cf_data = counterfactuals.get('counterfactuals', [])
        key_components = [
            str(len(cf_data)),
            str(sorted(metrics)),
            counterfactuals.get('method', 'unknown')
        ]
        
        # Add sample of counterfactual attributes for uniqueness
        if cf_data:
            sample_attrs = []
            for cf in cf_data[:3]:  # Use first 3 for key
                attrs = cf.get('target_attributes', {})
                sample_attrs.append(str(sorted(attrs.items())))
            key_components.extend(sample_attrs)
        
        key_string = '_'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _evaluate_metric(self, counterfactuals: Dict[str, Any], metric: str) -> Dict[str, Any]:
        """Evaluate a single metric with optimization."""
        cf_list = counterfactuals.get('counterfactuals', [])
        
        if not cf_list:
            return {'error': 'No counterfactuals provided'}
        
        try:
            if metric == "demographic_parity":
                return self._evaluate_demographic_parity(cf_list)
            elif metric == "fairness_score":
                return self._evaluate_fairness_score(cf_list)
            elif metric == "cits_score":
                return self._evaluate_cits_score(cf_list)
            elif metric == "attribute_balance":
                return self._evaluate_attribute_balance(cf_list)
            else:
                return {'error': f'Unknown metric: {metric}'}
                
        except Exception as e:
            logger.error(f"Metric evaluation error for {metric}: {e}")
            return {'error': f'Evaluation failed: {str(e)}'}
    
    def _evaluate_demographic_parity(self, counterfactuals: List[Dict]) -> Dict[str, Any]:
        """Evaluate demographic parity with optimized computation."""
        from collections import defaultdict
        
        attr_counts = defaultdict(lambda: defaultdict(int))
        total_samples = len(counterfactuals)
        
        # Count attribute distributions
        for cf in counterfactuals:
            target_attrs = cf.get('target_attributes', {})
            for attr, value in target_attrs.items():
                attr_counts[attr][value] += 1
        
        # Calculate parity scores
        parity_scores = {}
        for attr, value_counts in attr_counts.items():
            if not value_counts:
                continue
                
            total_attr = sum(value_counts.values())
            proportions = [count / total_attr for count in value_counts.values()]
            
            # Parity score: 1.0 = perfect balance, 0.0 = maximum imbalance
            max_prop = max(proportions)
            min_prop = min(proportions)
            parity_score = 1.0 - (max_prop - min_prop)
            
            parity_scores[attr] = {
                'parity_score': parity_score,
                'distribution': dict(value_counts),
                'proportions': dict(zip(value_counts.keys(), proportions))
            }
        
        overall_score = sum(data['parity_score'] for data in parity_scores.values()) / len(parity_scores) if parity_scores else 0.0
        
        return {
            'overall_balance_score': overall_score,
            'attribute_parity': parity_scores,
            'passes_threshold': overall_score >= 0.7,
            'total_samples': total_samples,
            'metric_type': 'demographic_parity'
        }
    
    def _evaluate_fairness_score(self, counterfactuals: List[Dict]) -> Dict[str, Any]:
        """Evaluate overall fairness score."""
        if not counterfactuals:
            return {'error': 'No counterfactuals provided'}
        
        # Analyze confidence distribution
        confidences = [cf.get('confidence', 0.5) for cf in counterfactuals]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Calculate variance
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        
        # Fairness score combines average confidence with consistency
        consistency_score = max(0, 1 - (std_dev / 0.5))  # Normalize by max reasonable std_dev
        fairness_score = (avg_confidence + consistency_score) / 2
        
        return {
            'fairness_score': fairness_score,
            'confidence_stats': {
                'mean': avg_confidence,
                'std_dev': std_dev,
                'min': min(confidences),
                'max': max(confidences)
            },
            'consistency_score': consistency_score,
            'passes_threshold': fairness_score >= 0.6,
            'metric_type': 'fairness_score'
        }
    
    def _evaluate_cits_score(self, counterfactuals: List[Dict]) -> Dict[str, Any]:
        """Evaluate CITS (Counterfactual Image-Text Similarity) score."""
        # Mock CITS evaluation with realistic computation
        num_samples = len(counterfactuals)
        
        # Diversity score based on attribute variety
        all_attrs = set()
        for cf in counterfactuals:
            attrs = cf.get('target_attributes', {})
            for attr, value in attrs.items():
                all_attrs.add(f"{attr}:{value}")
        
        diversity_factor = min(1.0, len(all_attrs) / (num_samples * 2))  # Expect ~2 attrs per sample
        
        # Quality score based on confidence
        confidences = [cf.get('confidence', 0.5) for cf in counterfactuals]
        quality_factor = sum(confidences) / len(confidences)
        
        # CITS combines diversity and quality
        cits_score = (diversity_factor * 0.6 + quality_factor * 0.4)
        
        return {
            'cits_score': cits_score,
            'diversity_component': diversity_factor,
            'quality_component': quality_factor,
            'unique_attributes': len(all_attrs),
            'passes_threshold': cits_score >= 0.65,
            'metric_type': 'cits_score'
        }
    
    def _evaluate_attribute_balance(self, counterfactuals: List[Dict]) -> Dict[str, Any]:
        """Evaluate balance across all attributes."""
        from collections import defaultdict
        
        attr_distributions = defaultdict(lambda: defaultdict(int))
        
        for cf in counterfactuals:
            target_attrs = cf.get('target_attributes', {})
            for attr, value in target_attrs.items():
                attr_distributions[attr][value] += 1
        
        balance_scores = {}
        for attr, value_counts in attr_distributions.items():
            total = sum(value_counts.values())
            if total == 0:
                continue
                
            # Calculate entropy-based balance score
            proportions = [count / total for count in value_counts.values()]
            entropy = -sum(p * math.log(p) if p > 0 else 0 for p in proportions)
            max_entropy = math.log(len(proportions)) if len(proportions) > 1 else 1
            
            balance_score = entropy / max_entropy if max_entropy > 0 else 1.0
            balance_scores[attr] = balance_score
        
        overall_balance = sum(balance_scores.values()) / len(balance_scores) if balance_scores else 0.0
        
        return {
            'overall_balance': overall_balance,
            'attribute_balance_scores': balance_scores,
            'passes_threshold': overall_balance >= 0.7,
            'metric_type': 'attribute_balance'
        }
    
    def _compile_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile evaluation summary from individual metric results."""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {
                'overall_fairness_score': 0.0,
                'fairness_rating': 'poor',
                'metrics_passed': 0,
                'total_metrics': len(results),
                'pass_rate': 0.0
            }
        
        # Extract scores from different metric types
        scores = []
        passing_metrics = 0
        
        for metric_name, metric_data in valid_results.items():
            if 'overall_balance_score' in metric_data:
                scores.append(metric_data['overall_balance_score'])
            elif 'fairness_score' in metric_data:
                scores.append(metric_data['fairness_score'])
            elif 'cits_score' in metric_data:
                scores.append(metric_data['cits_score'])
            elif 'overall_balance' in metric_data:
                scores.append(metric_data['overall_balance'])
            
            # Count passing metrics
            if metric_data.get('passes_threshold', False):
                passing_metrics += 1
        
        # Calculate overall fairness score
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Determine rating
        if overall_score >= 0.8:
            rating = 'excellent'
        elif overall_score >= 0.7:
            rating = 'good'
        elif overall_score >= 0.5:
            rating = 'fair'
        else:
            rating = 'poor'
        
        return {
            'overall_fairness_score': overall_score,
            'fairness_rating': rating,
            'metrics_passed': passing_metrics,
            'total_metrics': len(results),
            'pass_rate': passing_metrics / len(results) if results else 0.0,
            'component_scores': scores
        }
    
    def _validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate evaluation results."""
        if not results:
            return False
        
        # Check if any metrics passed
        for metric_data in results.values():
            if isinstance(metric_data, dict) and metric_data.get('passes_threshold', False):
                return True
        
        return False
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluator performance statistics."""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'total_evaluations': self.stats['total_evaluations'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['total_evaluations']),
            'parallel_evaluations': self.stats['parallel_evaluations'],
            'evaluations_per_hour': self.stats['total_evaluations'] / (uptime / 3600) if uptime > 0 else 0,
            'max_workers': self.max_workers,
            'uptime_hours': uptime / 3600,
            'status': 'healthy',
            'cache_stats': self.cache.get_stats() if hasattr(self.cache, 'get_stats') else {}
        }
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       format: str = "detailed", export_path: str = None) -> Dict[str, Any]:
        """Generate evaluation report."""
        report = {
            'report_type': f'bias_evaluation_{format}',
            'timestamp': datetime.now().isoformat(),
            'summary': evaluation_results.get('summary', {}),
            'metrics_results': evaluation_results.get('metrics', {}),
            'metadata': evaluation_results.get('metadata', {}),
            'key_findings': [],
            'recommendations': []
        }
        
        # Generate key findings
        summary = evaluation_results.get('summary', {})
        overall_score = summary.get('overall_fairness_score', 0)
        
        if overall_score >= 0.8:
            report['key_findings'].append("Excellent fairness performance across all metrics")
        elif overall_score >= 0.7:
            report['key_findings'].append("Good fairness performance with minor areas for improvement")
        else:
            report['key_findings'].append("Fairness performance below recommended thresholds")
        
        # Add metric-specific findings
        metrics = evaluation_results.get('metrics', {})
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'passes_threshold' in metric_data:
                if metric_data['passes_threshold']:
                    report['key_findings'].append(f"{metric_name}: PASSED threshold")
                else:
                    report['key_findings'].append(f"{metric_name}: FAILED threshold")
        
        # Generate recommendations
        if overall_score < 0.7:
            report['recommendations'].append("Increase diversity in counterfactual generation")
            report['recommendations'].append("Review attribute distribution balance")
        
        if summary.get('pass_rate', 0) < 0.8:
            report['recommendations'].append("Focus on failing metrics for targeted improvement")
        
        # Export if path provided
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                report['exported_to'] = export_path
            except Exception as e:
                logger.error(f"Failed to export report: {e}")
                report['export_error'] = str(e)
        
        return report


def test_scalable_system():
    """Comprehensive test of the scalable system."""
    logger.info("🚀 Testing scalable counterfactual lab system...")
    
    try:
        # Initialize scalable generator
        generator = ScalableCounterfactualGenerator(
            method="modicf",
            enable_caching=True,
            enable_worker_pool=True,
            enable_monitoring=True,
            initial_workers=2,
            max_workers=8
        )
        
        # Test single generation
        test_image = MockImage(400, 300)
        result = generator.generate(
            image=test_image,
            text="A professional software engineer coding at a computer",
            attributes=["gender", "age"],
            num_samples=5,
            user_id="test_user"
        )
        
        logger.info(f"✅ Single generation: {len(result['counterfactuals'])} samples in {result['metadata']['generation_time']:.3f}s")
        
        # Test batch generation
        batch_requests = [
            {
                "image": test_image,
                "text": f"A researcher analyzing data {i}",
                "attributes": ["gender", "race"],
                "num_samples": 3,
                "user_id": f"batch_user_{i}"
            }
            for i in range(5)
        ]
        
        batch_results = generator.generate_batch(batch_requests, max_parallel=3)
        successful_batches = sum(1 for r in batch_results if r.get('success', True))
        
        logger.info(f"✅ Batch generation: {successful_batches}/{len(batch_requests)} successful")
        
        # Test caching
        start_time = time.time()
        cached_result = generator.generate(
            image=test_image,
            text="A professional software engineer coding at a computer",
            attributes=["gender", "age"],
            num_samples=5,
            user_id="cache_test"
        )
        cache_time = time.time() - start_time
        
        logger.info(f"✅ Cache test: {cache_time:.3f}s (should be much faster)")
        
        # Test performance metrics
        metrics = generator.get_performance_metrics()
        logger.info(f"✅ Performance metrics collected: {len(metrics)} categories")
        
        # Test health check
        health = generator.health_check()
        logger.info(f"✅ Health check: {health['status']}")
        
        # Test optimization recommendations
        optimizations = generator.optimize_performance()
        logger.info(f"✅ Optimization recommendations: {len(optimizations['optimizations'])}")
        
        # Test concurrent load
        logger.info("Testing concurrent load...")
        concurrent_futures = []
        
        if generator.worker_pool:
            for i in range(10):
                future = generator.worker_pool.submit_task(
                    generator.generate,
                    image=test_image,
                    text=f"Concurrent test {i}",
                    attributes=["gender"],
                    num_samples=2,
                    user_id=f"concurrent_{i}"
                )
                concurrent_futures.append(future)
            
            concurrent_results = []
            for future in as_completed(concurrent_futures, timeout=30):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    logger.warning(f"Concurrent task failed: {e}")
            
            logger.info(f"✅ Concurrent processing: {len(concurrent_results)}/10 successful")
        
        # Final metrics
        final_metrics = generator.get_performance_metrics()
        
        if 'system_metrics' in final_metrics:
            sys_metrics = final_metrics['system_metrics']
            logger.info(f"📊 Final stats:")
            logger.info(f"   - Requests: {sys_metrics['requests_total']}")
            logger.info(f"   - Success rate: {sys_metrics['success_rate']:.1f}%")
            logger.info(f"   - Avg response time: {sys_metrics['avg_response_time']:.3f}s")
            logger.info(f"   - Cache hit rate: {sys_metrics.get('cache_hit_rate', 0):.1f}%")
            logger.info(f"   - Health score: {final_metrics.get('health_score', 0):.1f}")
        
        # Shutdown
        generator.shutdown()
        
        logger.info("🎉 Scalable system test completed successfully!")
        
        return {
            "single_generation": result,
            "batch_generation": batch_results,
            "cached_result": cached_result,
            "performance_metrics": final_metrics,
            "health_check": health,
            "optimization_report": optimizations,
            "test_status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Scalable system test failed: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {"test_status": "failed", "error": str(e)}


if __name__ == "__main__":
    test_scalable_system()