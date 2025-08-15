"""Enhanced error handling and logging system with self-healing integration."""

import logging
import traceback
import functools
import sys
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Type, List
from pathlib import Path
import json
from dataclasses import dataclass
from contextlib import contextmanager

from counterfactual_lab.exceptions import (
    GenerationError, ModelInitializationError, ValidationError,
    CacheError, StorageError, DeviceError
)


@dataclass
class ErrorContext:
    """Enhanced error context with self-healing information."""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: str
    component: str
    operation: str
    user_context: Dict[str, Any]
    system_state: Dict[str, Any]
    recovery_suggested: bool = False
    recovery_strategy: Optional[str] = None
    severity: str = "error"


class StructuredLogger:
    """Structured logging with JSON output and context tracking."""
    
    def __init__(self, 
                 name: str,
                 log_level: str = "INFO",
                 json_output: bool = True,
                 file_output: Optional[str] = None):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            json_output: Whether to use JSON formatting
            file_output: Optional file path for log output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if json_output:
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if file_output:
            file_handler = logging.FileHandler(file_output)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)
        
        # Context storage
        self._context = threading.local()
    
    def set_context(self, **context):
        """Set logging context for current thread."""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(context)
    
    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self._context, 'data'):
            self._context.data.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        if hasattr(self._context, 'data'):
            return self._context.data.copy()
        return {}
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context."""
        context = self.get_context()
        context.update(kwargs)
        
        extra = {
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context('CRITICAL', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context('DEBUG', message, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        # Add exception info if available
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)


class ErrorHandler:
    """Enhanced error handling with recovery suggestions."""
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """Initialize error handler.
        
        Args:
            logger: Optional structured logger instance
        """
        self.logger = logger or StructuredLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.recovery_suggestions: Dict[str, str] = {
            'ModelInitializationError': 'model_recovery',
            'DeviceError': 'model_recovery', 
            'CacheError': 'cache_recovery',
            'StorageError': 'storage_cleanup',
            'GenerationError': 'performance_optimization',
            'MemoryError': 'memory_pressure',
            'OutOfMemoryError': 'gpu_memory'
        }
        
        # Initialize error patterns
        self._initialize_error_patterns()
    
    def _initialize_error_patterns(self):
        """Initialize common error patterns and their characteristics."""
        self.error_patterns = {
            'memory_exhaustion': {
                'keywords': ['out of memory', 'memory', 'allocation failed'],
                'severity': 'critical',
                'recovery_strategy': 'memory_pressure',
                'auto_recoverable': True
            },
            'device_unavailable': {
                'keywords': ['cuda', 'device', 'gpu not available'],
                'severity': 'warning',
                'recovery_strategy': 'model_recovery',
                'auto_recoverable': True
            },
            'model_corruption': {
                'keywords': ['model', 'checkpoint', 'corrupted', 'invalid'],
                'severity': 'critical',
                'recovery_strategy': 'model_recovery',
                'auto_recoverable': False
            },
            'cache_corruption': {
                'keywords': ['cache', 'corrupted', 'invalid cache'],
                'severity': 'warning',
                'recovery_strategy': 'cache_recovery',
                'auto_recoverable': True
            },
            'storage_full': {
                'keywords': ['no space', 'disk full', 'storage'],
                'severity': 'critical',
                'recovery_strategy': 'storage_cleanup',
                'auto_recoverable': True
            },
            'network_timeout': {
                'keywords': ['timeout', 'network', 'connection'],
                'severity': 'warning',
                'recovery_strategy': 'performance_optimization',
                'auto_recoverable': True
            }
        }
    
    def handle_error(self,
                    error: Exception,
                    component: str,
                    operation: str,
                    user_context: Optional[Dict[str, Any]] = None,
                    system_state: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle error with enhanced context and recovery suggestions.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            user_context: User-specific context
            system_state: System state when error occurred
            
        Returns:
            ErrorContext with recovery information
        """
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Analyze error for patterns
        pattern_info = self._analyze_error_pattern(error_message, stack_trace)
        
        # Determine recovery strategy
        recovery_strategy = None
        recovery_suggested = False
        
        if error_type in self.recovery_suggestions:
            recovery_strategy = self.recovery_suggestions[error_type]
            recovery_suggested = True
        elif pattern_info:
            recovery_strategy = pattern_info.get('recovery_strategy')
            recovery_suggested = pattern_info.get('auto_recoverable', False)
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            timestamp=datetime.now().isoformat(),
            component=component,
            operation=operation,
            user_context=user_context or {},
            system_state=system_state or {},
            recovery_suggested=recovery_suggested,
            recovery_strategy=recovery_strategy,
            severity=pattern_info.get('severity', 'error') if pattern_info else 'error'
        )
        
        # Store in history
        self.error_history.append(error_context)
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Log error with context
        self.logger.set_context(
            component=component,
            operation=operation,
            error_type=error_type,
            recovery_suggested=recovery_suggested,
            recovery_strategy=recovery_strategy
        )
        
        if error_context.severity == 'critical':
            self.logger.critical(
                f"Critical error in {component}.{operation}: {error_message}",
                error_details=error_context.__dict__
            )
        else:
            self.logger.error(
                f"Error in {component}.{operation}: {error_message}",
                error_details=error_context.__dict__
            )
        
        return error_context
    
    def _analyze_error_pattern(self, error_message: str, stack_trace: str) -> Optional[Dict[str, Any]]:
        """Analyze error message and stack trace for known patterns."""
        error_text = (error_message + " " + stack_trace).lower()
        
        for pattern_name, pattern_info in self.error_patterns.items():
            keywords = pattern_info['keywords']
            if any(keyword in error_text for keyword in keywords):
                return pattern_info
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count by type
        error_types = {}
        components = {}
        recovery_suggested_count = 0
        critical_count = 0
        
        for error in self.error_history:
            # Count by type
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            # Count by component
            components[error.component] = components.get(error.component, 0) + 1
            
            # Count recovery suggestions
            if error.recovery_suggested:
                recovery_suggested_count += 1
            
            # Count critical errors
            if error.severity == 'critical':
                critical_count += 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "components": components,
            "recovery_suggested": recovery_suggested_count,
            "recovery_rate": recovery_suggested_count / len(self.error_history),
            "critical_errors": critical_count,
            "critical_rate": critical_count / len(self.error_history)
        }
    
    def get_recent_errors(self, hours: int = 24) -> List[ErrorContext]:
        """Get recent errors within specified hours."""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        return [
            error for error in self.error_history
            if datetime.fromisoformat(error.timestamp).timestamp() > cutoff_time
        ]
    
    def export_error_report(self, file_path: str):
        """Export error history and statistics to file."""
        try:
            report = {
                "export_timestamp": datetime.now().isoformat(),
                "statistics": self.get_error_statistics(),
                "error_history": [
                    {
                        "timestamp": error.timestamp,
                        "error_type": error.error_type,
                        "error_message": error.error_message,
                        "component": error.component,
                        "operation": error.operation,
                        "severity": error.severity,
                        "recovery_suggested": error.recovery_suggested,
                        "recovery_strategy": error.recovery_strategy,
                        "user_context": error.user_context,
                        "system_state": error.system_state
                    }
                    for error in self.error_history
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Error report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")


def with_error_handling(component: str, operation: str = None, 
                       error_handler: Optional[ErrorHandler] = None):
    """Decorator for enhanced error handling with recovery suggestions.
    
    Args:
        component: Component name where decorated function belongs
        operation: Operation name (defaults to function name)
        error_handler: Optional custom error handler
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            handler = error_handler or _get_global_error_handler()
            
            try:
                # Set logging context
                handler.logger.set_context(
                    component=component,
                    operation=op_name,
                    function=func.__name__
                )
                
                result = func(*args, **kwargs)
                
                # Log successful completion for critical operations
                if op_name in ['generate', 'initialize', 'load_model']:
                    handler.logger.info(f"Successfully completed {component}.{op_name}")
                
                return result
                
            except Exception as e:
                # Gather system state
                system_state = _gather_system_state()
                
                # Handle error
                error_context = handler.handle_error(
                    error=e,
                    component=component,
                    operation=op_name,
                    system_state=system_state
                )
                
                # Trigger self-healing if available and suggested
                if error_context.recovery_suggested:
                    _trigger_self_healing_if_available(error_context)
                
                # Re-raise the original exception
                raise
            
            finally:
                # Clear logging context
                handler.logger.clear_context()
        
        return wrapper
    return decorator


def _gather_system_state() -> Dict[str, Any]:
    """Gather current system state for error context."""
    try:
        import psutil
        
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if psutil.disk_usage('/') else None,
            "available_memory": psutil.virtual_memory().available,
            "process_count": len(psutil.pids())
        }
    except Exception:
        return {"error": "Failed to gather system state"}


def _trigger_self_healing_if_available(error_context: ErrorContext):
    """Trigger self-healing recovery if available."""
    try:
        from counterfactual_lab.self_healing_pipeline import get_global_guard
        
        guard = get_global_guard()
        if guard and error_context.recovery_strategy:
            # Create synthetic alert for recovery
            alert = {
                "type": "error_triggered_recovery",
                "message": f"Error-triggered recovery: {error_context.error_message}",
                "severity": error_context.severity,
                "value": 0,
                "error_context": error_context.__dict__
            }
            
            # Attempt recovery in background
            import threading
            recovery_thread = threading.Thread(
                target=lambda: guard._execute_recovery_for_alert_type(
                    "error_triggered_recovery", [alert]
                ),
                daemon=True
            )
            recovery_thread.start()
            
    except Exception:
        # Silently fail - self-healing is optional
        pass


@contextmanager
def error_boundary(component: str, operation: str,
                  error_handler: Optional[ErrorHandler] = None,
                  fallback_result: Any = None,
                  suppress_errors: bool = False):
    """Context manager for error boundaries with optional fallback.
    
    Args:
        component: Component name
        operation: Operation name
        error_handler: Optional custom error handler
        fallback_result: Result to return on error if suppress_errors=True
        suppress_errors: Whether to suppress errors and return fallback
    """
    handler = error_handler or _get_global_error_handler()
    
    try:
        # Set logging context
        handler.logger.set_context(
            component=component,
            operation=operation
        )
        
        yield
        
    except Exception as e:
        # Handle error
        system_state = _gather_system_state()
        error_context = handler.handle_error(
            error=e,
            component=component,
            operation=operation,
            system_state=system_state
        )
        
        # Trigger self-healing if suggested
        if error_context.recovery_suggested:
            _trigger_self_healing_if_available(error_context)
        
        if suppress_errors:
            # Return fallback result instead of raising
            return fallback_result
        else:
            raise
    
    finally:
        # Clear logging context
        handler.logger.clear_context()


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def _get_global_error_handler() -> ErrorHandler:
    """Internal helper to get global error handler."""
    return get_global_error_handler()

def initialize_error_handling(log_level: str = "INFO",
                            json_output: bool = True,
                            file_output: Optional[str] = None) -> ErrorHandler:
    """Initialize global error handling system.
    
    Args:
        log_level: Logging level
        json_output: Whether to use JSON formatting
        file_output: Optional file path for log output
        
    Returns:
        Global error handler instance
    """
    global _global_error_handler
    
    logger = StructuredLogger(
        name="counterfactual_lab",
        log_level=log_level,
        json_output=json_output,
        file_output=file_output
    )
    
    _global_error_handler = ErrorHandler(logger)
    return _global_error_handler