"""Performance optimization utilities for counterfactual generation."""

import logging
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import time
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from PIL import Image
import threading
from queue import Queue, Empty
import psutil

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    max_workers: int = None  # Auto-detect if None
    batch_size: int = 4
    use_gpu_batching: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    cache_preprocessed_images: bool = True
    optimize_image_sizes: bool = True
    target_image_size: Tuple[int, int] = (512, 512)
    memory_limit_mb: int = 8192  # 8GB default


class ImageProcessor:
    """Optimized image processing utilities."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._image_cache = {}
        self._cache_lock = threading.RLock()
        
    def preprocess_image(self, image: Union[str, Path, Image.Image], 
                        target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Preprocess image for optimal generation performance."""
        try:
            # Load image if needed
            if isinstance(image, (str, Path)):
                cache_key = str(image)
                
                # Check cache
                if self.config.cache_preprocessed_images and cache_key in self._image_cache:
                    return self._image_cache[cache_key].copy()
                
                pil_image = Image.open(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")
                cache_key = None
            
            # Optimize size
            if self.config.optimize_image_sizes:
                target_size = target_size or self.config.target_image_size
                pil_image = self._optimize_image_size(pil_image, target_size)
            
            # Apply preprocessing optimizations
            pil_image = self._apply_preprocessing_optimizations(pil_image)
            
            # Cache if enabled
            if self.config.cache_preprocessed_images and cache_key:
                with self._cache_lock:
                    self._image_cache[cache_key] = pil_image.copy()
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _optimize_image_size(self, image: Image.Image, 
                           target_size: Tuple[int, int]) -> Image.Image:
        """Optimize image size for processing."""
        current_size = image.size
        
        # Calculate optimal size maintaining aspect ratio
        aspect_ratio = current_size[0] / current_size[1]
        target_aspect = target_size[0] / target_size[1]
        
        if aspect_ratio > target_aspect:
            # Width is limiting factor
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            # Height is limiting factor
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)
        
        # Ensure minimum size
        new_width = max(64, new_width)
        new_height = max(64, new_height)
        
        # Only resize if significantly different
        if abs(current_size[0] - new_width) > 32 or abs(current_size[1] - new_height) > 32:
            logger.debug(f"Resizing image from {current_size} to ({new_width}, {new_height})")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _apply_preprocessing_optimizations(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing optimizations."""
        if self.config.enable_memory_optimization:
            # Convert to more memory-efficient format if needed
            if image.mode not in ["RGB", "L"]:
                image = image.convert("RGB")
        
        return image
    
    def batch_preprocess_images(self, images: List[Union[str, Path, Image.Image]]) -> List[Image.Image]:
        """Preprocess multiple images efficiently."""
        if not self.config.enable_parallel_processing or len(images) <= 2:
            return [self.preprocess_image(img) for img in images]
        
        # Use parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            processed_images = list(executor.map(self.preprocess_image, images))
        
        return processed_images
    
    def clear_cache(self):
        """Clear image preprocessing cache."""
        with self._cache_lock:
            self._image_cache.clear()
            logger.info("Image preprocessing cache cleared")


class BatchProcessor:
    """Handles batch processing of counterfactual generation requests."""
    
    def __init__(self, generator, config: OptimizationConfig):
        self.generator = generator
        self.config = config
        self.image_processor = ImageProcessor(config)
        
    def process_batch(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of generation requests."""
        if len(batch_requests) == 1:
            # Single request, process normally
            req = batch_requests[0]
            return [self.generator.generate(**req)]
        
        # Preprocess all images
        images = []
        for req in batch_requests:
            processed_image = self.image_processor.preprocess_image(req['image'])
            req['image'] = processed_image
            images.append(processed_image)
        
        if self.config.use_gpu_batching and self.generator.device == "cuda":
            return self._process_gpu_batch(batch_requests)
        else:
            return self._process_parallel_batch(batch_requests)
    
    def _process_gpu_batch(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch using GPU optimizations."""
        logger.info(f"Processing GPU batch of {len(batch_requests)} requests")
        
        # For now, process sequentially but with optimizations
        # In a real implementation, this would use GPU batch processing
        results = []
        
        try:
            for req in batch_requests:
                # Disable individual caching for batch processing
                req['save_results'] = False
                result = self.generator.generate(**req)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            # Fallback to parallel processing
            return self._process_parallel_batch(batch_requests)
    
    def _process_parallel_batch(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch using parallel processing."""
        logger.info(f"Processing parallel batch of {len(batch_requests)} requests")
        
        if not self.config.enable_parallel_processing:
            return [self.generator.generate(**req) for req in batch_requests]
        
        # Use thread-based parallelism for I/O bound tasks
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self.generator.generate, **req) for req in batch_requests]
            results = [future.result() for future in futures]
        
        return results
    
    def create_batches(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split requests into optimal batches."""
        batch_size = self.config.batch_size
        batches = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches from {len(requests)} requests")
        return batches


class MemoryManager:
    """Manages memory usage during generation."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_limit = config.memory_limit_mb * 1024 * 1024  # Convert to bytes
        
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        
        return {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "within_limit": memory.used < self.memory_limit
        }
    
    def optimize_memory_usage(self):
        """Optimize memory usage."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except ImportError:
            pass
        
        logger.debug("Memory optimization completed")
    
    def monitor_memory(self, operation_name: str):
        """Context manager for memory monitoring."""
        return MemoryMonitor(operation_name, self)


class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, operation_name: str, memory_manager: MemoryManager):
        self.operation_name = operation_name
        self.memory_manager = memory_manager
        self.start_memory = None
        
    def __enter__(self):
        self.start_memory = psutil.virtual_memory().used
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = psutil.virtual_memory().used
        memory_delta = end_memory - self.start_memory
        
        logger.debug(f"{self.operation_name} memory usage: {memory_delta / (1024*1024):.1f} MB")
        
        # Optimize if memory usage is high
        if end_memory > self.memory_manager.memory_limit:
            logger.warning("Memory limit exceeded, optimizing...")
            self.memory_manager.optimize_memory_usage()


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, generator, config: Optional[OptimizationConfig] = None):
        self.generator = generator
        self.config = config or self._create_default_config()
        
        self.batch_processor = BatchProcessor(generator, self.config)
        self.memory_manager = MemoryManager(self.config)
        
        logger.info(f"Performance optimizer initialized with config: {self.config}")
    
    def _create_default_config(self) -> OptimizationConfig:
        """Create default optimization configuration."""
        cpu_count = psutil.cpu_count(logical=False) or 4
        max_workers = min(cpu_count, 8)  # Cap at 8 to avoid overhead
        
        return OptimizationConfig(
            max_workers=max_workers,
            batch_size=min(4, max_workers),
            use_gpu_batching=self.generator.device == "cuda",
            enable_memory_optimization=True,
            enable_parallel_processing=True
        )
    
    def optimize_generation_request(self, **kwargs) -> Dict[str, Any]:
        """Optimize a single generation request."""
        with self.memory_manager.monitor_memory("single_generation"):
            # Preprocess image
            if 'image' in kwargs:
                kwargs['image'] = self.batch_processor.image_processor.preprocess_image(
                    kwargs['image']
                )
            
            return self.generator.generate(**kwargs)
    
    def optimize_batch_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize multiple generation requests."""
        if not requests:
            return []
        
        logger.info(f"Optimizing batch of {len(requests)} requests")
        
        with self.memory_manager.monitor_memory("batch_generation"):
            # Create optimal batches
            batches = self.batch_processor.create_batches(requests)
            
            # Process batches
            all_results = []
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)}")
                
                batch_results = self.batch_processor.process_batch(batch)
                all_results.extend(batch_results)
                
                # Optimize memory between batches
                if i < len(batches) - 1:  # Not the last batch
                    self.memory_manager.optimize_memory_usage()
        
        logger.info(f"Completed optimization of {len(requests)} requests")
        return all_results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        memory_info = self.memory_manager.check_memory_usage()
        
        return {
            "config": {
                "max_workers": self.config.max_workers,
                "batch_size": self.config.batch_size,
                "gpu_batching": self.config.use_gpu_batching,
                "parallel_processing": self.config.enable_parallel_processing
            },
            "memory": {
                "usage_mb": memory_info["used"] / (1024 * 1024),
                "available_mb": memory_info["available"] / (1024 * 1024),
                "percent_used": memory_info["percent"],
                "within_limit": memory_info["within_limit"]
            },
            "system": {
                "cpu_count": psutil.cpu_count(),
                "cpu_usage": psutil.cpu_percent(),
                "device": self.generator.device
            }
        }
    
    def update_config(self, **kwargs):
        """Update optimization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Reinitialize components if needed
        self.batch_processor = BatchProcessor(self.generator, self.config)
        self.memory_manager = MemoryManager(self.config)


class AsyncOptimizer:
    """Asynchronous optimization for high-throughput scenarios."""
    
    def __init__(self, generator, config: OptimizationConfig):
        self.generator = generator
        self.config = config
        self.optimizer = PerformanceOptimizer(generator, config)
        
    async def generate_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronous generation."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, 
                self.optimizer.optimize_generation_request,
                **kwargs
            )
        
        return result
    
    async def generate_batch_async(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Asynchronous batch generation."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = await loop.run_in_executor(
                executor,
                self.optimizer.optimize_batch_requests,
                requests
            )
        
        return results
    
    async def generate_concurrent(self, requests: List[Dict[str, Any]], 
                                max_concurrent: int = 4) -> List[Dict[str, Any]]:
        """Generate multiple requests concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_generate(request):
            async with semaphore:
                return await self.generate_async(**request)
        
        # Create tasks
        tasks = [bounded_generate(req) for req in requests]
        
        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                # You might want to add a placeholder or retry logic here
            else:
                successful_results.append(result)
        
        return successful_results


class CacheOptimizer:
    """Optimizes caching strategies for better performance."""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        
    def optimize_cache_strategy(self, usage_patterns: Dict[str, Any]):
        """Optimize cache based on usage patterns."""
        if not self.cache_manager:
            return
        
        # Analyze patterns and adjust cache settings
        hit_rate = usage_patterns.get("hit_rate", 0)
        
        if hit_rate < 0.3:  # Low hit rate
            logger.info("Low cache hit rate detected, cleaning up cache")
            self.cache_manager.cleanup_expired()
        
        # Implement more sophisticated cache optimization
        self._adjust_cache_size(usage_patterns)
        self._optimize_cache_ttl(usage_patterns)
    
    def _adjust_cache_size(self, patterns: Dict[str, Any]):
        """Adjust cache size based on usage."""
        # This is a placeholder - implement based on cache_manager capabilities
        pass
    
    def _optimize_cache_ttl(self, patterns: Dict[str, Any]):
        """Optimize cache TTL based on patterns."""
        # This is a placeholder - implement based on cache_manager capabilities
        pass