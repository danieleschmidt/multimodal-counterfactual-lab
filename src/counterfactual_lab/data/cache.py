"""Caching manager for performance optimization."""

import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for expensive operations like model inference and generation."""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 1000, ttl_hours: int = 24):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_hours * 3600
        
        # Create subdirectories for different cache types
        self.generations_dir = self.cache_dir / "generations"
        self.evaluations_dir = self.cache_dir / "evaluations"
        self.models_dir = self.cache_dir / "models"
        
        for dir_path in [self.generations_dir, self.evaluations_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Metadata file to track cache entries
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Cache manager initialized: {self.cache_dir} (max: {max_size_mb}MB, TTL: {ttl_hours}h)")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            "entries": {},
            "total_size": 0,
            "last_cleanup": datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """Generate a cache key from operation and parameters."""
        # Create a deterministic hash from the operation and parameters
        key_data = {
            "operation": operation,
            **kwargs
        }
        
        # Sort keys for consistency
        sorted_data = json.dumps(key_data, sort_keys=True)
        hash_obj = hashlib.sha256(sorted_data.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 characters
    
    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Get the file path for a cache entry."""
        if cache_type == "generation":
            return self.generations_dir / f"{key}.json"
        elif cache_type == "evaluation":
            return self.evaluations_dir / f"{key}.json"
        elif cache_type == "model":
            return self.models_dir / f"{key}.json"
        else:
            return self.cache_dir / f"{key}.json"
    
    def _is_expired(self, entry_time: str) -> bool:
        """Check if a cache entry is expired."""
        try:
            entry_datetime = datetime.fromisoformat(entry_time)
            return datetime.now() - entry_datetime > timedelta(seconds=self.ttl_seconds)
        except Exception:
            return True  # Consider invalid timestamps as expired
    
    def get(self, cache_type: str, operation: str, **kwargs) -> Optional[Any]:
        """Get cached result for an operation.
        
        Args:
            cache_type: Type of cache ("generation", "evaluation", "model")
            operation: Operation name
            **kwargs: Operation parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_cache_key(operation, **kwargs)
        cache_path = self._get_cache_path(cache_type, key)
        
        # Check metadata first
        if key not in self.metadata["entries"]:
            return None
        
        entry_info = self.metadata["entries"][key]
        
        # Check if expired
        if self._is_expired(entry_info["timestamp"]):
            logger.info(f"Cache entry expired: {key}")
            self._remove_entry(key)
            return None
        
        # Check if file exists
        if not cache_path.exists():
            logger.warning(f"Cache file missing: {cache_path}")
            self._remove_entry(key)
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Cache hit: {cache_type}/{operation} ({key})")
            return data
            
        except Exception as e:
            logger.error(f"Failed to read cache entry {key}: {e}")
            self._remove_entry(key)
            return None
    
    def set(self, cache_type: str, operation: str, result: Any, **kwargs) -> bool:
        """Store result in cache.
        
        Args:
            cache_type: Type of cache ("generation", "evaluation", "model")
            operation: Operation name
            result: Result to cache
            **kwargs: Operation parameters
            
        Returns:
            True if cached successfully, False otherwise
        """
        key = self._generate_cache_key(operation, **kwargs)
        cache_path = self._get_cache_path(cache_type, key)
        
        try:
            # Serialize result
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Update metadata
            file_size = cache_path.stat().st_size
            
            # Remove old entry if exists
            if key in self.metadata["entries"]:
                old_size = self.metadata["entries"][key]["size"]
                self.metadata["total_size"] -= old_size
            
            # Add new entry
            self.metadata["entries"][key] = {
                "cache_type": cache_type,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "size": file_size,
                "path": str(cache_path)
            }
            
            self.metadata["total_size"] += file_size
            self._save_metadata()
            
            logger.info(f"Cached: {cache_type}/{operation} ({key}, {file_size} bytes)")
            
            # Check if cleanup is needed
            if self.metadata["total_size"] > self.max_size_bytes:
                self._cleanup_lru()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache result for {operation}: {e}")
            return False
    
    def _remove_entry(self, key: str):
        """Remove a cache entry."""
        if key in self.metadata["entries"]:
            entry = self.metadata["entries"][key]
            
            # Remove file
            try:
                Path(entry["path"]).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
            
            # Update metadata
            self.metadata["total_size"] -= entry["size"]
            del self.metadata["entries"][key]
            self._save_metadata()
    
    def _cleanup_lru(self):
        """Cleanup cache using LRU strategy."""
        logger.info("Starting cache cleanup (LRU)")
        
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self.metadata["entries"].items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # Remove oldest entries until under size limit
        target_size = int(self.max_size_bytes * 0.8)  # Clean to 80% of max
        
        for key, entry in sorted_entries:
            if self.metadata["total_size"] <= target_size:
                break
            
            logger.info(f"Removing LRU cache entry: {key}")
            self._remove_entry(key)
        
        self.metadata["last_cleanup"] = datetime.now().isoformat()
        logger.info(f"Cache cleanup completed. Size: {self.metadata['total_size']} bytes")
    
    def cleanup_expired(self):
        """Remove all expired cache entries."""
        logger.info("Cleaning up expired cache entries")
        
        expired_keys = []
        for key, entry in self.metadata["entries"].items():
            if self._is_expired(entry["timestamp"]):
                expired_keys.append(key)
        
        for key in expired_keys:
            logger.info(f"Removing expired cache entry: {key}")
            self._remove_entry(key)
        
        logger.info(f"Removed {len(expired_keys)} expired entries")
    
    def clear_cache_type(self, cache_type: str):
        """Clear all entries of a specific cache type."""
        logger.info(f"Clearing cache type: {cache_type}")
        
        keys_to_remove = []
        for key, entry in self.metadata["entries"].items():
            if entry.get("cache_type") == cache_type:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_entry(key)
        
        logger.info(f"Cleared {len(keys_to_remove)} entries of type {cache_type}")
    
    def clear_all(self):
        """Clear all cache entries."""
        logger.info("Clearing all cache")
        
        # Remove all files
        for entry in self.metadata["entries"].values():
            try:
                Path(entry["path"]).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
        
        # Reset metadata
        self.metadata = {
            "entries": {},
            "total_size": 0,
            "last_cleanup": datetime.now().isoformat()
        }
        self._save_metadata()
        
        logger.info("All cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_entries": len(self.metadata["entries"]),
            "total_size_bytes": self.metadata["total_size"],
            "total_size_mb": round(self.metadata["total_size"] / (1024 * 1024), 2),
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "utilization_percent": round(
                (self.metadata["total_size"] / self.max_size_bytes) * 100, 1
            ),
            "last_cleanup": self.metadata["last_cleanup"],
            "ttl_hours": self.ttl_seconds / 3600
        }
        
        # Count by cache type
        type_counts = {}
        type_sizes = {}
        
        for entry in self.metadata["entries"].values():
            cache_type = entry.get("cache_type", "unknown")
            type_counts[cache_type] = type_counts.get(cache_type, 0) + 1
            type_sizes[cache_type] = type_sizes.get(cache_type, 0) + entry["size"]
        
        stats["by_type"] = {
            "counts": type_counts,
            "sizes_mb": {k: round(v / (1024 * 1024), 2) for k, v in type_sizes.items()}
        }
        
        return stats
    
    def cache_generation_result(self, method: str, image_hash: str, text: str, 
                              attributes: Dict[str, str], result: Any) -> bool:
        """Cache counterfactual generation result."""
        return self.set(
            "generation", 
            "counterfactual_generation",
            result,
            method=method,
            image_hash=image_hash,
            text=text,
            attributes=attributes
        )
    
    def get_cached_generation(self, method: str, image_hash: str, text: str, 
                            attributes: Dict[str, str]) -> Optional[Any]:
        """Get cached counterfactual generation result."""
        return self.get(
            "generation",
            "counterfactual_generation",
            method=method,
            image_hash=image_hash,
            text=text,
            attributes=attributes
        )
    
    def cache_evaluation_result(self, model_name: str, counterfactual_hash: str,
                              metrics: List[str], result: Any) -> bool:
        """Cache bias evaluation result."""
        return self.set(
            "evaluation",
            "bias_evaluation", 
            result,
            model_name=model_name,
            counterfactual_hash=counterfactual_hash,
            metrics=sorted(metrics)  # Sort for consistency
        )
    
    def get_cached_evaluation(self, model_name: str, counterfactual_hash: str,
                            metrics: List[str]) -> Optional[Any]:
        """Get cached bias evaluation result."""
        return self.get(
            "evaluation",
            "bias_evaluation",
            model_name=model_name,
            counterfactual_hash=counterfactual_hash,
            metrics=sorted(metrics)
        )