"""Scalable core with performance optimization, caching, and concurrent processing."""

import json
import logging
import random
import hashlib
import traceback
import time
import asyncio
import threading
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