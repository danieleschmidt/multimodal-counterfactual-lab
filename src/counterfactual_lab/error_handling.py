"""Advanced error handling and recovery mechanisms."""

import logging
import traceback
import time
import functools
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
import json

from counterfactual_lab.exceptions import (
    CounterfactualLabError, GenerationError, ModelInitializationError,
    ValidationError, CacheError, StorageError, EvaluationError,
    AttributeError, DeviceError, ConfigurationError
)

logger = logging.getLogger(__name__)


class ErrorContext:
    """Context information for error tracking."""
    
    def __init__(self, operation: str, user_id: Optional[str] = None, **kwargs):
        self.operation = operation
        self.user_id = user_id
        self.context = kwargs
        self.start_time = time.time()
        self.error_count = 0
    
    def add_context(self, **kwargs):
        """Add additional context information."""
        self.context.update(kwargs)
    
    def increment_error(self):
        """Increment error counter."""
        self.error_count += 1
    
    def get_duration(self) -> float:
        """Get operation duration in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'operation': self.operation,
            'user_id': self.user_id,
            'duration': self.get_duration(),
            'error_count': self.error_count,
            'context': self.context,
            'timestamp': datetime.now().isoformat()
        }


class ErrorRecoveryStrategies:
    """Error recovery and retry strategies."""
    
    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 300.0) -> float:
        """Calculate exponential backoff delay."""
        delay = min(base_delay * (2 ** attempt), max_delay)
        return delay
    
    @staticmethod
    def linear_backoff(attempt: int, base_delay: float = 1.0, increment: float = 1.0) -> float:
        """Calculate linear backoff delay."""
        return base_delay + (increment * attempt)
    
    @staticmethod
    def should_retry(exception: Exception, attempt: int, max_attempts: int = 3) -> bool:
        """Determine if operation should be retried."""
        # Don't retry validation errors or configuration errors
        if isinstance(exception, (ValidationError, ConfigurationError)):
            return False
        
        # Don't retry if max attempts reached
        if attempt >= max_attempts:
            return False
        
        # Retry for transient errors
        transient_errors = (
            CacheError, StorageError, DeviceError, 
            ConnectionError, TimeoutError
        )
        
        return isinstance(exception, transient_errors)


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, name: str = "default"):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        
        logger.info(f"Circuit breaker '{name}' initialized: threshold={failure_threshold}, timeout={timeout}s")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
            else:
                raise GenerationError(f"Circuit breaker '{self.name}' is open")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.reset()
                logger.info(f"Circuit breaker '{self.name}' reset after successful call")
            
            return result
            
        except Exception as e:
            self.record_failure()
            raise
    
    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def reset(self):
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'timeout': self.timeout
        }


class ErrorTracker:
    """Comprehensive error tracking and analytics."""
    
    def __init__(self, storage_file: str = "error_logs.json"):
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self._load_error_history()
    
    def _load_error_history(self):
        """Load error history from file."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.error_history = data.get('error_history', [])
                    self.error_counts = data.get('error_counts', {})
                    logger.info(f"Loaded {len(self.error_history)} error records")
            except Exception as e:
                logger.error(f"Failed to load error history: {e}")
    
    def _save_error_history(self):
        """Save error history to file."""
        try:
            data = {
                'error_history': self.error_history[-1000:],  # Keep last 1000 errors
                'error_counts': self.error_counts,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save error history: {e}")
    
    def record_error(self, error: Exception, context: ErrorContext):
        """Record an error occurrence."""
        error_type = type(error).__name__
        error_info = {
            'error_type': error_type,
            'error_message': str(error),
            'operation': context.operation,
            'user_id': context.user_id,
            'duration': context.get_duration(),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc(),
            'context': context.context
        }
        
        self.error_history.append(error_info)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Periodic save (every 10 errors)
        if len(self.error_history) % 10 == 0:
            self._save_error_history()
        
        logger.error(f"Error recorded: {error_type} in {context.operation}")
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error['timestamp']) > cutoff_time
        ]
        
        stats = {
            'total_errors': len(recent_errors),
            'error_types': {},
            'operations': {},
            'users': {},
            'time_period_hours': hours
        }
        
        for error in recent_errors:
            # Count by type
            error_type = error['error_type']
            stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
            
            # Count by operation
            operation = error['operation']
            stats['operations'][operation] = stats['operations'].get(operation, 0) + 1
            
            # Count by user
            user_id = error.get('user_id', 'unknown')
            stats['users'][user_id] = stats['users'].get(user_id, 0) + 1
        
        return stats
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        
        return self.circuit_breakers[name]
    
    def get_all_circuit_breaker_status(self) -> List[Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return [cb.get_status() for cb in self.circuit_breakers.values()]


def with_error_handling(
    operation_name: str,
    max_retries: int = 3,
    backoff_strategy: str = "exponential",
    circuit_breaker_name: Optional[str] = None,
    fallback_function: Optional[Callable] = None
):
    """Decorator for comprehensive error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get error tracker from global context or create one
            error_tracker = getattr(wrapper, '_error_tracker', None)
            if error_tracker is None:
                error_tracker = ErrorTracker()
                wrapper._error_tracker = error_tracker
            
            context = ErrorContext(operation_name, user_id=kwargs.get('user_id'))
            
            # Get circuit breaker if specified
            circuit_breaker = None
            if circuit_breaker_name:
                circuit_breaker = error_tracker.get_circuit_breaker(circuit_breaker_name)
            
            def execute_with_circuit_breaker():
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    context.add_context(attempt=attempt)
                    result = execute_with_circuit_breaker()
                    
                    # Log successful recovery if this wasn't the first attempt
                    if attempt > 0:
                        logger.info(f"Operation {operation_name} succeeded after {attempt} retries")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    context.increment_error()
                    error_tracker.record_error(e, context)
                    
                    # Check if we should retry
                    if attempt < max_retries and ErrorRecoveryStrategies.should_retry(e, attempt, max_retries):
                        # Calculate backoff delay
                        if backoff_strategy == "exponential":
                            delay = ErrorRecoveryStrategies.exponential_backoff(attempt)
                        else:
                            delay = ErrorRecoveryStrategies.linear_backoff(attempt)
                        
                        logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                        time.sleep(delay)
                        continue
                    
                    # Max retries reached or non-retryable error
                    logger.error(f"Operation {operation_name} failed after {attempt + 1} attempts: {e}")
                    
                    # Try fallback function
                    if fallback_function:
                        try:
                            logger.info(f"Attempting fallback for operation {operation_name}")
                            return fallback_function(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")
                    
                    # Re-raise the last exception
                    raise last_exception
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator


class HealthChecker:
    """System health checking and monitoring."""
    
    def __init__(self, error_tracker: ErrorTracker):
        self.error_tracker = error_tracker
        self.health_checks: Dict[str, Callable] = {}
        self.health_status = {}
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {'status': 'unknown', 'error': 'Health check not found'}
        
        try:
            result = self.health_checks[name]()
            
            if isinstance(result, bool):
                status = 'healthy' if result else 'unhealthy'
                result = {'status': status}
            elif not isinstance(result, dict):
                result = {'status': 'healthy', 'data': result}
            
            result['last_check'] = datetime.now().isoformat()
            self.health_status[name] = result
            
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
            self.health_status[name] = error_result
            logger.error(f"Health check '{name}' failed: {e}")
            return error_result
    
    def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.health_checks:
            results[name] = self.run_health_check(name)
        
        return results
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        health_results = self.run_all_health_checks()
        error_stats = self.error_tracker.get_error_statistics(hours=1)  # Last hour
        circuit_breaker_status = self.error_tracker.get_all_circuit_breaker_status()
        
        # Calculate overall health score
        total_checks = len(health_results)
        healthy_checks = sum(1 for result in health_results.values() if result.get('status') == 'healthy')
        health_score = healthy_checks / total_checks if total_checks > 0 else 1.0
        
        # Determine overall status
        if health_score >= 0.9:
            overall_status = 'healthy'
        elif health_score >= 0.7:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        # Check for circuit breaker issues
        open_circuits = sum(1 for cb in circuit_breaker_status if cb['state'] == 'open')
        if open_circuits > 0:
            overall_status = 'degraded'
        
        return {
            'overall_status': overall_status,
            'health_score': health_score,
            'timestamp': datetime.now().isoformat(),
            'health_checks': health_results,
            'error_statistics': error_stats,
            'circuit_breakers': circuit_breaker_status,
            'open_circuit_breakers': open_circuits,
            'recommendations': self._generate_recommendations(health_results, error_stats, circuit_breaker_status)
        }
    
    def _generate_recommendations(self, health_results: Dict, error_stats: Dict, circuit_status: List) -> List[str]:
        """Generate health recommendations based on current status."""
        recommendations = []
        
        # Check for unhealthy components
        unhealthy_checks = [name for name, result in health_results.items() if result.get('status') != 'healthy']
        if unhealthy_checks:
            recommendations.append(f"Investigate unhealthy components: {', '.join(unhealthy_checks)}")
        
        # Check error rates
        if error_stats.get('total_errors', 0) > 10:  # More than 10 errors in last hour
            recommendations.append("High error rate detected in the last hour")
        
        # Check circuit breakers
        open_circuits = [cb['name'] for cb in circuit_status if cb['state'] == 'open']
        if open_circuits:
            recommendations.append(f"Circuit breakers are open: {', '.join(open_circuits)}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")
        
        return recommendations


# Global error tracker instance
_global_error_tracker = ErrorTracker()

def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _global_error_tracker