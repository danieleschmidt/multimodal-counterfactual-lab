"""Production-ready core with full scaling, security, and monitoring."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import threading

from counterfactual_lab.enhanced_core import EnhancedCounterfactualGenerator, EnhancedBiasEvaluator
from counterfactual_lab.advanced_optimization import (
    ScalingConfig, AsyncGenerationManager, DistributedGenerationCluster
)
from counterfactual_lab.error_handling import with_error_handling, HealthChecker, get_error_tracker
from counterfactual_lab.security import AuditLogger, SecurityMiddleware, SecureSessionManager
from counterfactual_lab.monitoring import SystemDiagnostics
from counterfactual_lab.exceptions import GenerationError, SecurityError

logger = logging.getLogger(__name__)


class ProductionCounterfactualGenerator:
    """Production-ready counterfactual generator with full enterprise features."""
    
    def __init__(
        self,
        method: str = "modicf",
        device: str = "auto",
        enable_security: bool = True,
        enable_async: bool = True,
        enable_distributed: bool = False,
        scaling_config: Optional[ScalingConfig] = None,
        **kwargs
    ):
        """Initialize production generator with enterprise features."""
        
        self.method = method
        self.device = device
        self.enable_security = enable_security
        self.enable_async = enable_async
        self.enable_distributed = enable_distributed
        
        # Configuration
        self.scaling_config = scaling_config or ScalingConfig()
        
        # Core components
        self.base_generator = EnhancedCounterfactualGenerator(
            method=method,
            device=device,
            enable_security=enable_security,
            **kwargs
        )
        
        # Advanced features
        if enable_async:
            self.async_manager = AsyncGenerationManager(
                self.base_generator, 
                self.scaling_config
            )
        else:
            self.async_manager = None
        
        if enable_distributed:
            self.distributed_cluster = DistributedGenerationCluster(
                self.scaling_config
            )
        else:
            self.distributed_cluster = None
        
        # Monitoring and diagnostics
        self.system_diagnostics = SystemDiagnostics()
        self.error_tracker = get_error_tracker()
        self.audit_logger = AuditLogger("production_audit.log")
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'requests_per_minute': 0.0,
            'peak_concurrent_users': 0,
            'uptime_start': datetime.now()
        }
        
        # Request tracking
        self._active_requests = {}
        self._request_history = []
        self._metrics_lock = threading.Lock()
        
        # Health monitoring thread
        self._health_monitor_thread = None
        self._start_health_monitoring()
        
        logger.info(
            f"Production generator initialized: "
            f"async={enable_async}, distributed={enable_distributed}, "
            f"security={enable_security}"
        )
    
    def _start_health_monitoring(self):
        """Start continuous health monitoring."""
        def health_monitor_loop():
            while True:
                try:
                    self._update_performance_metrics()
                    self._cleanup_old_data()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(60)
        
        self._health_monitor_thread = threading.Thread(
            target=health_monitor_loop, 
            daemon=True
        )
        self._health_monitor_thread.start()
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        with self._metrics_lock:
            # Calculate requests per minute
            current_time = datetime.now()
            one_minute_ago = current_time - timedelta(minutes=1)
            
            recent_requests = [
                req for req in self._request_history
                if req['timestamp'] > one_minute_ago
            ]
            
            self.performance_metrics['requests_per_minute'] = len(recent_requests)
            self.performance_metrics['peak_concurrent_users'] = max(
                self.performance_metrics['peak_concurrent_users'],
                len(self._active_requests)
            )
    
    def _cleanup_old_data(self):
        """Clean up old performance data."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._metrics_lock:
            self._request_history = [
                req for req in self._request_history
                if req['timestamp'] > cutoff_time
            ]
    
    def _record_request(self, request_id: str, success: bool, duration: float, user_id: str = None):
        """Record request for performance tracking."""
        with self._metrics_lock:
            self.performance_metrics['total_requests'] += 1
            
            if success:
                self.performance_metrics['successful_requests'] += 1
            else:
                self.performance_metrics['failed_requests'] += 1
            
            # Update average response time
            if self.performance_metrics['total_requests'] == 1:
                self.performance_metrics['avg_response_time'] = duration
            else:
                current_avg = self.performance_metrics['avg_response_time']
                new_avg = (current_avg + duration) / 2
                self.performance_metrics['avg_response_time'] = new_avg
            
            # Add to history
            self._request_history.append({
                'request_id': request_id,
                'timestamp': datetime.now(),
                'success': success,
                'duration': duration,
                'user_id': user_id
            })
    
    @with_error_handling("production_generate", max_retries=3, circuit_breaker_name="production")
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        text: str,
        attributes: Union[List[str], str],
        num_samples: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate counterfactuals with full production features."""
        
        request_id = f"req_{int(time.time() * 1000)}_{id(self)}"
        start_time = time.time()
        
        try:
            # Add to active requests
            self._active_requests[request_id] = {
                'start_time': start_time,
                'user_id': user_id,
                'method': self.method,
                'attributes': attributes,
                'num_samples': num_samples
            }
            
            # Security audit
            if self.enable_security:
                self.audit_logger.log_generation_request(
                    user_id or 'anonymous',
                    self.method,
                    attributes if isinstance(attributes, list) else [attributes],
                    True  # Will update if fails
                )
            
            # Route to appropriate processing method
            if self.enable_distributed and self.distributed_cluster:
                result = self._generate_distributed(
                    image=image, text=text, attributes=attributes,
                    num_samples=num_samples, user_id=user_id, **kwargs
                )
            elif self.enable_async and self.async_manager:
                # For sync interface to async backend
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.async_manager.generate_async(
                            image=image, text=text, attributes=attributes,
                            num_samples=num_samples, user_id=user_id, **kwargs
                        )
                    )
                finally:
                    loop.close()
            else:
                # Direct processing
                result = self.base_generator.secure_generate(
                    image=image, text=text, attributes=attributes,
                    num_samples=num_samples, user_id=user_id, **kwargs
                )
            
            # Add production metadata
            result['production_metadata'] = {
                'request_id': request_id,
                'processing_method': self._get_processing_method(),
                'generation_timestamp': datetime.now().isoformat(),
                'priority': priority,
                'version': '1.0.0'
            }
            
            duration = time.time() - start_time
            self._record_request(request_id, True, duration, user_id)
            
            logger.info(f"Production generation completed: {request_id} ({duration:.2f}s)")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_request(request_id, False, duration, user_id)
            
            logger.error(f"Production generation failed: {request_id} ({duration:.2f}s): {e}")
            raise
            
        finally:
            # Remove from active requests
            if request_id in self._active_requests:
                del self._active_requests[request_id]
    
    def _get_processing_method(self) -> str:
        """Get the current processing method."""
        if self.enable_distributed:
            return "distributed"
        elif self.enable_async:
            return "async"
        else:
            return "direct"
    
    def _generate_distributed(self, **params) -> Dict[str, Any]:
        """Generate using distributed cluster."""
        if not self.distributed_cluster:
            raise GenerationError("Distributed processing not available")
        
        # Submit to cluster
        task_id = self.distributed_cluster.submit_distributed_task(**params)
        
        # Wait for result
        result = self.distributed_cluster.get_distributed_result(task_id, timeout=300)
        
        if not result['success']:
            raise GenerationError(f"Distributed generation failed: {result.get('error', 'Unknown error')}")
        
        return result['result']
    
    async def generate_async(self, **params) -> Dict[str, Any]:
        """Asynchronous generation interface."""
        if not self.async_manager:
            raise GenerationError("Async processing not enabled")
        
        return await self.async_manager.generate_async(**params)
    
    async def generate_batch_async(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Asynchronous batch generation."""
        if not self.async_manager:
            raise GenerationError("Async processing not enabled")
        
        return await self.async_manager.generate_batch_async(requests)
    
    def generate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous batch generation with optimization."""
        logger.info(f"Processing batch of {len(requests)} requests")
        
        start_time = time.time()
        
        if self.enable_async and len(requests) > 1:
            # Use async processing for batches
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    self.generate_batch_async(requests)
                )
            finally:
                loop.close()
        else:
            # Sequential processing
            results = []
            for i, request in enumerate(requests):
                logger.info(f"Processing batch request {i+1}/{len(requests)}")
                try:
                    result = self.generate(**request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch request {i+1} failed: {e}")
                    results.append({
                        'error': str(e),
                        'success': False,
                        'request_index': i
                    })
        
        duration = time.time() - start_time
        logger.info(f"Batch processing completed: {len(requests)} requests in {duration:.2f}s")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.performance_metrics['uptime_start']).total_seconds(),
            'configuration': {
                'method': self.method,
                'device': self.device,
                'security_enabled': self.enable_security,
                'async_enabled': self.enable_async,
                'distributed_enabled': self.enable_distributed
            },
            'performance_metrics': self.performance_metrics.copy(),
            'active_requests': len(self._active_requests),
            'health_status': self.base_generator.get_health_status()
        }
        
        # Add async manager stats
        if self.async_manager:
            status['async_stats'] = self.async_manager.get_performance_stats()
        
        # Add distributed cluster stats
        if self.distributed_cluster and self.enable_distributed:
            status['distributed_stats'] = {
                'worker_processes': len(self.distributed_cluster.worker_processes),
                'enabled': True
            }
        
        # Add system diagnostics
        try:
            status['system_diagnostics'] = self.system_diagnostics.run_full_diagnostics()
        except Exception as e:
            status['system_diagnostics'] = {'error': str(e)}
        
        return status
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""
        with self._metrics_lock:
            total_requests = self.performance_metrics['total_requests']
            successful_requests = self.performance_metrics['successful_requests']
            
            success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
            
            # Request distribution over time
            recent_requests = self._request_history[-100:]  # Last 100 requests
            
            # Calculate percentiles
            durations = [req['duration'] for req in recent_requests]
            durations.sort()
            
            percentiles = {}
            if durations:
                percentiles = {
                    'p50': durations[int(len(durations) * 0.5)],
                    'p90': durations[int(len(durations) * 0.9)],
                    'p95': durations[int(len(durations) * 0.95)],
                    'p99': durations[int(len(durations) * 0.99)]
                }
            
        return {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_requests': total_requests,
                'success_rate': success_rate,
                'avg_response_time_seconds': self.performance_metrics['avg_response_time'],
                'requests_per_minute': self.performance_metrics['requests_per_minute'],
                'peak_concurrent_users': self.performance_metrics['peak_concurrent_users']
            },
            'response_time_percentiles': percentiles,
            'error_statistics': self.error_tracker.get_error_statistics(hours=24),
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check success rate
        total = self.performance_metrics['total_requests']
        successful = self.performance_metrics['successful_requests']
        success_rate = successful / total if total > 0 else 1.0
        
        if success_rate < 0.95:
            recommendations.append("Success rate below 95% - investigate error patterns")
        
        # Check response time
        avg_time = self.performance_metrics['avg_response_time']
        if avg_time > 10.0:
            recommendations.append("Average response time > 10s - consider enabling async processing")
        
        # Check concurrent users
        peak_concurrent = self.performance_metrics['peak_concurrent_users']
        if peak_concurrent > 50 and not self.enable_async:
            recommendations.append("High concurrent usage detected - enable async processing")
        
        # Check if distributed processing could help
        if peak_concurrent > 100 and not self.enable_distributed:
            recommendations.append("Very high concurrent usage - consider distributed processing")
        
        if not recommendations:
            recommendations.append("System performance appears optimal")
        
        return recommendations
    
    def shutdown(self):
        """Gracefully shutdown the production generator."""
        logger.info("Initiating production generator shutdown...")
        
        # Wait for active requests to complete (with timeout)
        timeout = 30.0
        start_time = time.time()
        
        while self._active_requests and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for {len(self._active_requests)} active requests to complete...")
            time.sleep(1)
        
        if self._active_requests:
            logger.warning(f"Timeout reached, {len(self._active_requests)} requests still active")
        
        # Shutdown distributed cluster
        if self.distributed_cluster:
            self.distributed_cluster.shutdown_cluster()
        
        # Final audit log
        self.audit_logger.log_security_event(
            'system_shutdown', 'info',
            'Production generator shutdown completed',
            {'final_metrics': self.performance_metrics}
        )
        
        logger.info("Production generator shutdown completed")


class ProductionBiasEvaluator:
    """Production-ready bias evaluator with enterprise features."""
    
    def __init__(self, model, enable_security: bool = True, **kwargs):
        """Initialize production evaluator."""
        
        self.base_evaluator = EnhancedBiasEvaluator(
            model, 
            enable_security=enable_security,
            **kwargs
        )
        
        self.audit_logger = AuditLogger("evaluation_audit.log")
        self.performance_metrics = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'avg_evaluation_time': 0.0
        }
        
        logger.info("Production bias evaluator initialized")
    
    @with_error_handling("production_evaluate", max_retries=2)
    def evaluate(
        self,
        counterfactuals: Dict[str, Any],
        metrics: List[str],
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate bias with production features."""
        
        evaluation_id = f"eval_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            result = self.base_evaluator.secure_evaluate(
                counterfactuals=counterfactuals,
                metrics=metrics,
                user_id=user_id
            )
            
            # Add production metadata
            result['production_metadata'] = {
                'evaluation_id': evaluation_id,
                'evaluation_timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            duration = time.time() - start_time
            self._record_evaluation(True, duration)
            
            logger.info(f"Production evaluation completed: {evaluation_id} ({duration:.2f}s)")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_evaluation(False, duration)
            
            logger.error(f"Production evaluation failed: {evaluation_id} ({duration:.2f}s): {e}")
            raise
    
    def _record_evaluation(self, success: bool, duration: float):
        """Record evaluation metrics."""
        self.performance_metrics['total_evaluations'] += 1
        
        if success:
            self.performance_metrics['successful_evaluations'] += 1
        
        # Update average time
        if self.performance_metrics['total_evaluations'] == 1:
            self.performance_metrics['avg_evaluation_time'] = duration
        else:
            current_avg = self.performance_metrics['avg_evaluation_time']
            new_avg = (current_avg + duration) / 2
            self.performance_metrics['avg_evaluation_time'] = new_avg
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        total = self.performance_metrics['total_evaluations']
        successful = self.performance_metrics['successful_evaluations']
        
        return {
            'total_evaluations': total,
            'successful_evaluations': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_evaluation_time': self.performance_metrics['avg_evaluation_time']
        }