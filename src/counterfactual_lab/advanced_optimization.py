"""Advanced optimization and scaling features for production deployment."""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import weakref
import gc
import sys
from collections import defaultdict, deque
import hashlib
import numpy as np
from PIL import Image

from counterfactual_lab.optimization import OptimizationConfig, PerformanceOptimizer
from counterfactual_lab.error_handling import with_error_handling, CircuitBreaker
from counterfactual_lab.exceptions import GenerationError, DeviceError

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for scaling and advanced optimizations."""
    # Async processing
    max_concurrent_requests: int = 10
    async_batch_size: int = 5
    async_timeout: float = 300.0  # 5 minutes
    
    # Distributed processing
    enable_distributed: bool = False
    worker_processes: int = None  # Auto-detect
    shared_memory_size: int = 1024 * 1024 * 100  # 100MB
    
    # Advanced caching
    enable_ml_cache: bool = True
    cache_similarity_threshold: float = 0.95
    cache_compression: bool = True
    
    # Load balancing
    enable_load_balancing: bool = True
    worker_health_check_interval: float = 30.0
    max_queue_size: int = 1000
    
    # Memory optimization
    enable_memory_pool: bool = True
    memory_pool_size: int = 1024 * 1024 * 500  # 500MB
    enable_garbage_collection: bool = True
    gc_threshold_mb: int = 1000


class AdvancedImageProcessor:
    """GPU-accelerated and optimized image processing."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._gpu_available = self._check_gpu_availability()
        self._batch_cache = {}
        self._preprocessing_pipeline = None
        self._init_preprocessing_pipeline()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _init_preprocessing_pipeline(self):
        """Initialize optimized preprocessing pipeline."""
        if self._gpu_available:
            try:
                import torch
                import torchvision.transforms as transforms
                
                self._preprocessing_pipeline = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                logger.info("GPU-accelerated preprocessing pipeline initialized")
                
            except ImportError:
                logger.warning("GPU available but PyTorch not installed")
        
        # Fallback CPU pipeline
        if self._preprocessing_pipeline is None:
            self._preprocessing_pipeline = self._cpu_preprocessing_pipeline
    
    def _cpu_preprocessing_pipeline(self, image: Image.Image) -> np.ndarray:
        """CPU-based preprocessing pipeline."""
        # Resize
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Channel normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        return img_array
    
    def process_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Process a batch of images efficiently."""
        if self._gpu_available and len(images) > 1:
            return self._gpu_batch_process(images)
        else:
            return [self._cpu_preprocessing_pipeline(img) for img in images]
    
    def _gpu_batch_process(self, images: List[Image.Image]) -> List[np.ndarray]:
        """GPU-accelerated batch processing."""
        try:
            import torch
            
            # Stack images into batch tensor
            batch_tensor = torch.stack([
                self._preprocessing_pipeline(img) for img in images
            ]).cuda()
            
            # Move back to CPU and convert to numpy
            batch_numpy = batch_tensor.cpu().numpy()
            
            return [batch_numpy[i] for i in range(len(images))]
            
        except Exception as e:
            logger.warning(f"GPU batch processing failed, falling back to CPU: {e}")
            return [self._cpu_preprocessing_pipeline(img) for img in images]


class IntelligentCache:
    """ML-powered intelligent caching with similarity detection."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.cache = {}
        self.embeddings = {}  # Store embeddings for similarity comparison
        self._hit_count = 0
        self._miss_count = 0
        self._similarity_index = defaultdict(list)
    
    def _compute_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Compute image embedding for similarity comparison."""
        # Simple perceptual hash - in production would use deep learning embeddings
        image_small = image.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
        pixels = np.array(image_small, dtype=np.float32)
        
        # DCT-based hash
        dct = self._simple_dct(pixels)
        hash_bits = dct > np.median(dct)
        
        return hash_bits.astype(np.float32)
    
    def _simple_dct(self, image: np.ndarray) -> np.ndarray:
        """Simple 2D DCT implementation."""
        # Simplified DCT for demonstration
        return np.fft.fft2(image).real
    
    def _compute_text_embedding(self, text: str) -> np.ndarray:
        """Compute text embedding for similarity comparison."""
        # Simple bag-of-words embedding
        words = text.lower().split()
        
        # Create a simple vocabulary hash
        vocab_hash = {}
        for word in words:
            word_hash = hash(word) % 1000
            vocab_hash[word_hash] = vocab_hash.get(word_hash, 0) + 1
        
        # Create fixed-size embedding vector
        embedding = np.zeros(100)
        for word_hash, count in vocab_hash.items():
            embedding[word_hash % 100] += count
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_similar_result(self, image: Image.Image, text: str, attributes: List[str]) -> Optional[Any]:
        """Get cached result based on similarity."""
        if not self.config.enable_ml_cache:
            return None
        
        # Compute embeddings
        img_embedding = self._compute_image_embedding(image)
        text_embedding = self._compute_text_embedding(text)
        
        # Find similar entries
        best_similarity = 0.0
        best_result = None
        
        for cache_key, (cached_result, cached_embeddings) in self.cache.items():
            img_sim = self._compute_similarity(img_embedding, cached_embeddings['image'])
            text_sim = self._compute_similarity(text_embedding, cached_embeddings['text'])
            
            # Combined similarity score
            combined_sim = (img_sim + text_sim) / 2.0
            
            if combined_sim > best_similarity and combined_sim > self.config.cache_similarity_threshold:
                # Check if attributes match
                cached_attrs = cached_embeddings.get('attributes', [])
                if set(attributes) == set(cached_attrs):
                    best_similarity = combined_sim
                    best_result = cached_result
        
        if best_result:
            self._hit_count += 1
            logger.debug(f"ML cache hit with similarity {best_similarity:.3f}")
        else:
            self._miss_count += 1
        
        return best_result
    
    def cache_result(self, image: Image.Image, text: str, attributes: List[str], result: Any):
        """Cache result with embeddings."""
        if not self.config.enable_ml_cache:
            return
        
        # Compute embeddings
        img_embedding = self._compute_image_embedding(image)
        text_embedding = self._compute_text_embedding(text)
        
        # Create cache key
        cache_key = hashlib.sha256(
            f"{text}{','.join(sorted(attributes))}".encode()
        ).hexdigest()
        
        # Store with embeddings
        embeddings = {
            'image': img_embedding,
            'text': text_embedding,
            'attributes': attributes
        }
        
        # Compress if enabled
        if self.config.cache_compression:
            try:
                import gzip
                result = gzip.compress(pickle.dumps(result))
            except Exception:
                pass  # Fall back to uncompressed
        
        self.cache[cache_key] = (result, embeddings)
        
        # Limit cache size
        if len(self.cache) > 10000:
            # Remove oldest entries (simplified LRU)
            oldest_keys = list(self.cache.keys())[:1000]
            for key in oldest_keys:
                del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'enabled': self.config.enable_ml_cache
        }


class LoadBalancer:
    """Intelligent load balancing for distributed processing."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.workers = {}
        self.worker_stats = defaultdict(lambda: {'requests': 0, 'errors': 0, 'avg_time': 0.0})
        self.request_queue = queue.Queue(maxsize=config.max_queue_size)
        self._health_check_thread = None
        self._start_health_monitoring()
    
    def _start_health_monitoring(self):
        """Start health monitoring for workers."""
        if self.config.enable_load_balancing:
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop, daemon=True
            )
            self._health_check_thread.start()
    
    def _health_check_loop(self):
        """Continuous health checking of workers."""
        while True:
            try:
                self._check_worker_health()
                time.sleep(self.config.worker_health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def _check_worker_health(self):
        """Check health of all workers."""
        unhealthy_workers = []
        
        for worker_id, worker in self.workers.items():
            try:
                # Simple health check - ping worker
                if hasattr(worker, 'ping') and not worker.ping():
                    unhealthy_workers.append(worker_id)
            except Exception:
                unhealthy_workers.append(worker_id)
        
        # Remove unhealthy workers
        for worker_id in unhealthy_workers:
            logger.warning(f"Removing unhealthy worker: {worker_id}")
            del self.workers[worker_id]
            if worker_id in self.worker_stats:
                del self.worker_stats[worker_id]
    
    def select_worker(self) -> Optional[str]:
        """Select best available worker based on load and performance."""
        if not self.workers:
            return None
        
        # Calculate worker scores based on performance
        worker_scores = {}
        
        for worker_id in self.workers:
            stats = self.worker_stats[worker_id]
            
            # Score based on: fewer requests, fewer errors, faster response
            request_score = 1.0 / (stats['requests'] + 1)
            error_score = 1.0 / (stats['errors'] + 1)
            time_score = 1.0 / (stats['avg_time'] + 0.1)
            
            combined_score = request_score * error_score * time_score
            worker_scores[worker_id] = combined_score
        
        # Select worker with highest score
        best_worker = max(worker_scores.items(), key=lambda x: x[1])
        return best_worker[0]
    
    def submit_request(self, request: Dict[str, Any]) -> str:
        """Submit request for processing."""
        if self.request_queue.full():
            raise GenerationError("Request queue is full")
        
        request_id = hashlib.sha256(
            f"{time.time()}{id(request)}".encode()
        ).hexdigest()[:16]
        
        request['id'] = request_id
        request['timestamp'] = time.time()
        
        self.request_queue.put(request)
        return request_id
    
    def record_worker_stats(self, worker_id: str, success: bool, duration: float):
        """Record worker performance statistics."""
        stats = self.worker_stats[worker_id]
        stats['requests'] += 1
        
        if not success:
            stats['errors'] += 1
        
        # Update average processing time
        if stats['requests'] == 1:
            stats['avg_time'] = duration
        else:
            stats['avg_time'] = (stats['avg_time'] + duration) / 2
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'active_workers': len(self.workers),
            'queue_size': self.request_queue.qsize(),
            'max_queue_size': self.config.max_queue_size,
            'worker_stats': dict(self.worker_stats),
            'enabled': self.config.enable_load_balancing
        }


class AsyncGenerationManager:
    """Manages asynchronous counterfactual generation with advanced optimizations."""
    
    def __init__(self, generator, config: ScalingConfig):
        self.generator = generator
        self.config = config
        
        # Initialize components
        self.image_processor = AdvancedImageProcessor(config)
        self.intelligent_cache = IntelligentCache(config)
        self.load_balancer = LoadBalancer(config)
        
        # Async coordination
        self.active_tasks = {}
        self.task_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Performance monitoring
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'peak_concurrent_requests': 0
        }
        
        logger.info(f"AsyncGenerationManager initialized with max_concurrent={config.max_concurrent_requests}")
    
    async def generate_async(self, **request_params) -> Dict[str, Any]:
        """Asynchronous counterfactual generation."""
        async with self.task_semaphore:
            request_id = hashlib.sha256(
                f"{time.time()}{id(request_params)}".encode()
            ).hexdigest()[:16]
            
            start_time = time.time()
            
            try:
                # Update stats
                self.performance_stats['total_requests'] += 1
                current_concurrent = len(self.active_tasks)
                self.performance_stats['peak_concurrent_requests'] = max(
                    self.performance_stats['peak_concurrent_requests'],
                    current_concurrent
                )
                
                # Add to active tasks
                self.active_tasks[request_id] = {
                    'start_time': start_time,
                    'params': request_params
                }
                
                # Try intelligent cache first
                cache_result = self.intelligent_cache.get_similar_result(
                    request_params.get('image'),
                    request_params.get('text', ''),
                    request_params.get('attributes', [])
                )
                
                if cache_result:
                    self.performance_stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for request {request_id}")
                    return cache_result
                
                # Process request
                result = await self._process_request_async(**request_params)
                
                # Cache result
                self.intelligent_cache.cache_result(
                    request_params.get('image'),
                    request_params.get('text', ''),
                    request_params.get('attributes', []),
                    result
                )
                
                # Update stats
                self.performance_stats['successful_requests'] += 1
                duration = time.time() - start_time
                
                if self.performance_stats['total_requests'] == 1:
                    self.performance_stats['avg_processing_time'] = duration
                else:
                    self.performance_stats['avg_processing_time'] = (
                        self.performance_stats['avg_processing_time'] + duration
                    ) / 2
                
                return result
                
            except Exception as e:
                logger.error(f"Async generation failed for request {request_id}: {e}")
                raise
            
            finally:
                # Remove from active tasks
                if request_id in self.active_tasks:
                    del self.active_tasks[request_id]
    
    async def _process_request_async(self, **request_params) -> Dict[str, Any]:
        """Process individual request asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Use thread pool for CPU-bound operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            def run_generation():
                # Filter parameters to match base generator interface
                base_params = {
                    k: v for k, v in request_params.items()
                    if k in ['image', 'text', 'attributes', 'num_samples', 'save_results', 'experiment_id']
                }
                return self.generator.generate(**base_params)
            
            future = loop.run_in_executor(
                executor,
                run_generation
            )
            
            # Add timeout
            try:
                result = await asyncio.wait_for(
                    future, 
                    timeout=self.config.async_timeout
                )
                return result
            except asyncio.TimeoutError:
                raise GenerationError(f"Request timed out after {self.config.async_timeout}s")
    
    async def generate_batch_async(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple requests asynchronously."""
        logger.info(f"Processing async batch of {len(requests)} requests")
        
        # Create async tasks
        tasks = []
        for i, request in enumerate(requests):
            task = asyncio.create_task(
                self.generate_async(**request),
                name=f"batch_request_{i}"
            )
            tasks.append(task)
        
        # Process batches
        results = []
        batch_size = self.config.async_batch_size
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(
                *batch_tasks, 
                return_exceptions=True
            )
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch request failed: {result}")
                    results.append({
                        'error': str(result),
                        'success': False
                    })
                else:
                    results.append(result)
        
        logger.info(f"Async batch processing completed: {len(results)} results")
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'generation_stats': self.performance_stats.copy(),
            'cache_stats': self.intelligent_cache.get_stats(),
            'load_balancer_stats': self.load_balancer.get_stats(),
            'active_tasks': len(self.active_tasks),
            'image_processor_gpu': self.image_processor._gpu_available
        }
        
        # Add success rate
        if stats['generation_stats']['total_requests'] > 0:
            stats['generation_stats']['success_rate'] = (
                stats['generation_stats']['successful_requests'] /
                stats['generation_stats']['total_requests']
            )
        else:
            stats['generation_stats']['success_rate'] = 0.0
        
        return stats


class DistributedGenerationCluster:
    """Distributed generation cluster for maximum scalability."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.worker_processes = []
        self.master_process = True
        self.work_queue = None
        self.result_queue = None
        
        if config.enable_distributed:
            self._initialize_cluster()
    
    def _initialize_cluster(self):
        """Initialize distributed processing cluster."""
        try:
            # Create shared queues
            self.work_queue = mp.Queue(maxsize=self.config.max_queue_size)
            self.result_queue = mp.Queue()
            
            # Start worker processes
            num_workers = self.config.worker_processes or mp.cpu_count()
            
            for i in range(num_workers):
                worker = mp.Process(
                    target=self._worker_process,
                    args=(i, self.work_queue, self.result_queue),
                    daemon=True
                )
                worker.start()
                self.worker_processes.append(worker)
            
            logger.info(f"Distributed cluster initialized with {num_workers} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed cluster: {e}")
            self.config.enable_distributed = False
    
    def _worker_process(self, worker_id: int, work_queue: mp.Queue, result_queue: mp.Queue):
        """Worker process for distributed generation."""
        logger.info(f"Worker {worker_id} started")
        
        # Initialize generator in worker process
        from counterfactual_lab import CounterfactualGenerator
        generator = CounterfactualGenerator(device='cpu')  # Use CPU for workers
        
        while True:
            try:
                # Get work item
                work_item = work_queue.get(timeout=30)
                
                if work_item is None:  # Shutdown signal
                    break
                
                task_id = work_item['task_id']
                params = work_item['params']
                
                # Process request
                start_time = time.time()
                try:
                    result = generator.generate(**params)
                    result_queue.put({
                        'task_id': task_id,
                        'result': result,
                        'success': True,
                        'worker_id': worker_id,
                        'processing_time': time.time() - start_time
                    })
                except Exception as e:
                    result_queue.put({
                        'task_id': task_id,
                        'error': str(e),
                        'success': False,
                        'worker_id': worker_id,
                        'processing_time': time.time() - start_time
                    })
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                break
        
        logger.info(f"Worker {worker_id} shutting down")
    
    def submit_distributed_task(self, **params) -> str:
        """Submit task for distributed processing."""
        if not self.config.enable_distributed or not self.work_queue:
            raise GenerationError("Distributed processing not available")
        
        task_id = hashlib.sha256(
            f"{time.time()}{id(params)}".encode()
        ).hexdigest()[:16]
        
        work_item = {
            'task_id': task_id,
            'params': params,
            'timestamp': time.time()
        }
        
        try:
            self.work_queue.put(work_item, timeout=5)
            return task_id
        except queue.Full:
            raise GenerationError("Distributed work queue is full")
    
    def get_distributed_result(self, task_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Get result from distributed processing."""
        if not self.config.enable_distributed or not self.result_queue:
            raise GenerationError("Distributed processing not available")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result_item = self.result_queue.get(timeout=1)
                
                if result_item['task_id'] == task_id:
                    return result_item
                else:
                    # Put back result that belongs to another task
                    self.result_queue.put(result_item)
                    
            except queue.Empty:
                continue
        
        raise GenerationError(f"Distributed task {task_id} timed out")
    
    def shutdown_cluster(self):
        """Shutdown distributed cluster."""
        if not self.config.enable_distributed:
            return
        
        logger.info("Shutting down distributed cluster...")
        
        # Send shutdown signals
        if self.work_queue:
            for _ in self.worker_processes:
                self.work_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.worker_processes:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        
        logger.info("Distributed cluster shutdown completed")