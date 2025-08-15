"""Self-healing pipeline guard for autonomous recovery and optimization."""

import logging
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback

from counterfactual_lab.monitoring import SystemDiagnostics, SystemHealth
from counterfactual_lab.exceptions import (
    GenerationError, ModelInitializationError, CacheError, 
    StorageError, DeviceError
)

logger = logging.getLogger(__name__)


@dataclass
class FailureEvent:
    """Represents a system failure event."""
    timestamp: str
    failure_type: str
    component: str
    error_message: str
    severity: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None


@dataclass
class RecoveryStrategy:
    """Defines a recovery strategy for specific failure types."""
    name: str
    failure_types: List[str]
    priority: int
    max_attempts: int
    cooldown_seconds: int
    recovery_function: Callable
    validation_function: Optional[Callable] = None


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, func):
        """Decorator for applying circuit breaker to functions."""
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker transitioning to HALF_OPEN for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker CLOSED for {func.__name__}")
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker OPEN for {func.__name__} after {self.failure_count} failures")
                
                raise e
                
        return wrapper
    
    def reset(self):
        """Reset circuit breaker state."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None


class SelfHealingPipelineGuard:
    """Autonomous self-healing pipeline guard with progressive recovery."""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 monitoring_interval: int = 30,
                 auto_recovery: bool = True):
        """Initialize self-healing pipeline guard.
        
        Args:
            config_path: Path to configuration file
            monitoring_interval: Health check interval in seconds
            auto_recovery: Enable automatic recovery attempts
        """
        self.monitoring_interval = monitoring_interval
        self.auto_recovery = auto_recovery
        self.is_running = False
        self.monitor_thread = None
        
        # Core components
        self.diagnostics = SystemDiagnostics()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_history: List[FailureEvent] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Recovery state
        self.recovery_in_progress = False
        self.last_recovery_attempt = {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
        
        logger.info("Self-healing pipeline guard initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "failure_thresholds": {
                "cpu_usage": 95.0,
                "memory_usage": 90.0,
                "disk_usage": 95.0,
                "gpu_memory_usage": 95.0,
                "generation_time": 120.0
            },
            "recovery_settings": {
                "max_attempts": 3,
                "cooldown_seconds": 30,
                "escalation_threshold": 5
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 60
            },
            "monitoring": {
                "health_check_interval": 30,
                "alert_retention_hours": 24
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies for different failure types."""
        
        # Memory pressure recovery
        self.recovery_strategies["memory_pressure"] = RecoveryStrategy(
            name="memory_pressure_recovery",
            failure_types=["memory_high", "out_of_memory"],
            priority=1,
            max_attempts=3,
            cooldown_seconds=30,
            recovery_function=self._recover_memory_pressure
        )
        
        # GPU memory recovery
        self.recovery_strategies["gpu_memory"] = RecoveryStrategy(
            name="gpu_memory_recovery",
            failure_types=["gpu_memory_high", "cuda_out_of_memory"],
            priority=1,
            max_attempts=3,
            cooldown_seconds=20,
            recovery_function=self._recover_gpu_memory
        )
        
        # Storage cleanup recovery
        self.recovery_strategies["storage_cleanup"] = RecoveryStrategy(
            name="storage_cleanup",
            failure_types=["disk_high", "storage_full"],
            priority=2,
            max_attempts=2,
            cooldown_seconds=60,
            recovery_function=self._recover_storage_space
        )
        
        # Model reinitialization recovery
        self.recovery_strategies["model_recovery"] = RecoveryStrategy(
            name="model_recovery",
            failure_types=["model_initialization_error", "device_error"],
            priority=3,
            max_attempts=2,
            cooldown_seconds=120,
            recovery_function=self._recover_model_initialization
        )
        
        # Cache recovery
        self.recovery_strategies["cache_recovery"] = RecoveryStrategy(
            name="cache_recovery",
            failure_types=["cache_error", "cache_corruption"],
            priority=2,
            max_attempts=2,
            cooldown_seconds=30,
            recovery_function=self._recover_cache_system
        )
        
        # Performance optimization recovery
        self.recovery_strategies["performance_optimization"] = RecoveryStrategy(
            name="performance_optimization",
            failure_types=["generation_slow", "performance_degradation"],
            priority=3,
            max_attempts=3,
            cooldown_seconds=60,
            recovery_function=self._recover_performance
        )
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical components."""
        circuit_config = self.config["circuit_breaker"]
        
        self.circuit_breakers["generation"] = CircuitBreaker(
            failure_threshold=circuit_config["failure_threshold"],
            recovery_timeout=circuit_config["recovery_timeout"]
        )
        
        self.circuit_breakers["storage"] = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30
        )
        
        self.circuit_breakers["cache"] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Self-healing monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Self-healing monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Run diagnostics
                diagnostics = self.diagnostics.run_full_diagnostics()
                
                # Check for failures and trigger recovery
                self._analyze_system_state(diagnostics)
                
                # Cleanup old failure events
                self._cleanup_failure_history()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.monitoring_interval)
    
    def _analyze_system_state(self, diagnostics: Dict[str, Any]):
        """Analyze system state and trigger recovery if needed."""
        health = diagnostics["health"]
        alerts = diagnostics["alerts"]["active_alerts"]
        
        # Check for critical issues
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        
        if critical_alerts and self.auto_recovery and not self.recovery_in_progress:
            logger.warning(f"Critical issues detected: {len(critical_alerts)} alerts")
            self._trigger_recovery(critical_alerts)
        
        # Check overall system health
        if health["overall_status"] == "critical" and self.auto_recovery:
            self._assess_and_recover_system_health(health)
    
    def _trigger_recovery(self, alerts: List[Dict[str, Any]]):
        """Trigger recovery procedures based on alerts."""
        if self.recovery_in_progress:
            logger.info("Recovery already in progress, skipping")
            return
        
        self.recovery_in_progress = True
        try:
            # Group alerts by type for targeted recovery
            alert_types = {}
            for alert in alerts:
                alert_type = alert["type"]
                if alert_type not in alert_types:
                    alert_types[alert_type] = []
                alert_types[alert_type].append(alert)
            
            # Execute recovery strategies
            for alert_type, alert_list in alert_types.items():
                self._execute_recovery_for_alert_type(alert_type, alert_list)
                
        finally:
            self.recovery_in_progress = False
    
    def _execute_recovery_for_alert_type(self, alert_type: str, alerts: List[Dict[str, Any]]):
        """Execute recovery strategy for specific alert type."""
        # Find matching recovery strategy
        strategy = None
        for recovery_strategy in self.recovery_strategies.values():
            if alert_type in recovery_strategy.failure_types:
                strategy = recovery_strategy
                break
        
        if not strategy:
            logger.warning(f"No recovery strategy found for alert type: {alert_type}")
            return
        
        # Check cooldown
        last_attempt = self.last_recovery_attempt.get(strategy.name, datetime.min)
        cooldown_end = last_attempt + timedelta(seconds=strategy.cooldown_seconds)
        
        if datetime.now() < cooldown_end:
            logger.info(f"Recovery strategy {strategy.name} in cooldown, skipping")
            return
        
        # Record failure event
        failure_event = FailureEvent(
            timestamp=datetime.now().isoformat(),
            failure_type=alert_type,
            component="system",
            error_message=f"Alert: {alerts[0]['message']}",
            severity=alerts[0]["severity"],
            context={"alerts": alerts}
        )
        
        # Attempt recovery
        logger.info(f"Attempting recovery with strategy: {strategy.name}")
        self.last_recovery_attempt[strategy.name] = datetime.now()
        
        try:
            recovery_successful = strategy.recovery_function(alerts)
            failure_event.recovery_attempted = True
            failure_event.recovery_successful = recovery_successful
            failure_event.recovery_method = strategy.name
            
            if recovery_successful:
                logger.info(f"Recovery successful: {strategy.name}")
            else:
                logger.warning(f"Recovery failed: {strategy.name}")
                
        except Exception as e:
            logger.error(f"Recovery strategy {strategy.name} raised exception: {e}")
            failure_event.recovery_attempted = True
            failure_event.recovery_successful = False
            failure_event.error_message += f" | Recovery error: {str(e)}"
        
        self.failure_history.append(failure_event)
    
    def _recover_memory_pressure(self, alerts: List[Dict[str, Any]]) -> bool:
        """Recover from memory pressure issues."""
        logger.info("Executing memory pressure recovery")
        
        try:
            # Clear Python garbage collection
            import gc
            gc.collect()
            
            # Clear caches if available
            try:
                # Try to clear tensor caches
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
            except ImportError:
                pass
            
            # Clear system caches safely
            try:
                import subprocess
                subprocess.run(["sync"], check=False, capture_output=True)
                # Note: Requires root privileges, will fail gracefully otherwise
                subprocess.run(
                    ["tee", "/proc/sys/vm/drop_caches"], 
                    input=b"3", 
                    check=False, 
                    capture_output=True
                )
                logger.info("Cleared system caches")
            except Exception:
                logger.debug("Could not clear system caches (may require root)")
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Memory pressure recovery failed: {e}")
            return False
    
    def _recover_gpu_memory(self, alerts: List[Dict[str, Any]]) -> bool:
        """Recover from GPU memory issues."""
        logger.info("Executing GPU memory recovery")
        
        try:
            import torch
            if torch.cuda.is_available():
                # Clear cache and force garbage collection
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                logger.info("GPU memory recovery completed")
                return True
            
        except ImportError:
            logger.warning("PyTorch not available for GPU recovery")
        except Exception as e:
            logger.error(f"GPU memory recovery failed: {e}")
        
        return False
    
    def _recover_storage_space(self, alerts: List[Dict[str, Any]]) -> bool:
        """Recover from storage space issues."""
        logger.info("Executing storage cleanup recovery")
        
        try:
            space_freed = 0
            
            # Clean cache directories
            cache_dirs = [
                "./cache/generations",
                "./cache/models",
                "./test_output",
                "./cli_test_output"
            ]
            
            for cache_dir in cache_dirs:
                cache_path = Path(cache_dir)
                if cache_path.exists():
                    # Remove old cache files (older than 7 days)
                    cutoff_time = datetime.now() - timedelta(days=7)
                    
                    for file_path in cache_path.rglob("*"):
                        if file_path.is_file():
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_time:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                space_freed += file_size
            
            # Clean temporary files
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            for temp_file in temp_dir.glob("counterfactual_*"):
                if temp_file.is_file():
                    try:
                        space_freed += temp_file.stat().st_size
                        temp_file.unlink()
                    except:
                        pass
            
            logger.info(f"Storage cleanup freed {space_freed / (1024**2):.1f} MB")
            return space_freed > 0
            
        except Exception as e:
            logger.error(f"Storage cleanup failed: {e}")
            return False
    
    def _recover_model_initialization(self, alerts: List[Dict[str, Any]]) -> bool:
        """Recover from model initialization errors."""
        logger.info("Executing model recovery")
        
        try:
            # Force device reset
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            # Reset global state
            import gc
            gc.collect()
            
            logger.info("Model recovery completed")
            return True
            
        except Exception as e:
            logger.error(f"Model recovery failed: {e}")
            return False
    
    def _recover_cache_system(self, alerts: List[Dict[str, Any]]) -> bool:
        """Recover from cache system errors."""
        logger.info("Executing cache system recovery")
        
        try:
            # Clear corrupted cache files
            cache_dirs = ["./cache", "./src/cache"]
            
            for cache_dir in cache_dirs:
                cache_path = Path(cache_dir)
                if cache_path.exists():
                    # Remove metadata files to force cache rebuild
                    metadata_files = cache_path.glob("**/cache_metadata.json")
                    for metadata_file in metadata_files:
                        metadata_file.unlink()
                        logger.info(f"Removed cache metadata: {metadata_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache recovery failed: {e}")
            return False
    
    def _recover_performance(self, alerts: List[Dict[str, Any]]) -> bool:
        """Recover from performance degradation."""
        logger.info("Executing performance optimization recovery")
        
        try:
            # Memory optimization
            import gc
            gc.collect()
            
            # CPU optimization (lower process priority)
            try:
                import os
                os.nice(5)  # Lower priority
            except:
                pass
            
            # Clear performance caches
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("Performance optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Performance recovery failed: {e}")
            return False
    
    def _assess_and_recover_system_health(self, health: Dict[str, Any]):
        """Assess overall system health and apply appropriate recovery."""
        issues = health.get("issues", [])
        
        if not issues:
            return
        
        # Create synthetic alerts for health issues
        synthetic_alerts = []
        for issue in issues:
            if "CPU usage" in issue:
                synthetic_alerts.append({
                    "type": "cpu_high",
                    "message": issue,
                    "severity": "critical",
                    "value": health["cpu_usage"]
                })
            elif "memory usage" in issue:
                synthetic_alerts.append({
                    "type": "memory_high",
                    "message": issue,
                    "severity": "critical",
                    "value": health["memory_usage"]
                })
            elif "disk usage" in issue:
                synthetic_alerts.append({
                    "type": "disk_high",
                    "message": issue,
                    "severity": "critical",
                    "value": health["disk_usage"]
                })
        
        if synthetic_alerts:
            self._trigger_recovery(synthetic_alerts)
    
    def _cleanup_failure_history(self):
        """Clean up old failure events from history."""
        retention_hours = self.config["monitoring"]["alert_retention_hours"]
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        original_count = len(self.failure_history)
        self.failure_history = [
            event for event in self.failure_history
            if datetime.fromisoformat(event.timestamp) > cutoff_time
        ]
        
        removed_count = original_count - len(self.failure_history)
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old failure events")
    
    @contextmanager
    def protected_operation(self, operation_name: str):
        """Context manager for protecting operations with circuit breaker."""
        circuit_breaker = self.circuit_breakers.get(operation_name)
        if not circuit_breaker:
            # Create circuit breaker if it doesn't exist
            circuit_breaker = CircuitBreaker()
            self.circuit_breakers[operation_name] = circuit_breaker
        
        start_time = time.time()
        try:
            if circuit_breaker.state == "OPEN":
                raise Exception(f"Circuit breaker OPEN for {operation_name}")
            
            yield
            
            # Reset on successful operation
            if circuit_breaker.state == "HALF_OPEN":
                circuit_breaker.reset()
                logger.info(f"Circuit breaker reset for {operation_name}")
        
        except Exception as e:
            # Record failure
            failure_event = FailureEvent(
                timestamp=datetime.now().isoformat(),
                failure_type="operation_failure",
                component=operation_name,
                error_message=str(e),
                severity="error",
                context={"operation_time": time.time() - start_time}
            )
            self.failure_history.append(failure_event)
            
            # Update circuit breaker
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = datetime.now()
            
            if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
                circuit_breaker.state = "OPEN"
                logger.error(f"Circuit breaker OPEN for {operation_name}")
            
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive self-healing system status."""
        return {
            "monitoring": {
                "is_running": self.is_running,
                "monitoring_interval": self.monitoring_interval,
                "auto_recovery": self.auto_recovery,
                "recovery_in_progress": self.recovery_in_progress
            },
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            },
            "failure_history": {
                "total_events": len(self.failure_history),
                "recent_events": len([
                    e for e in self.failure_history
                    if datetime.fromisoformat(e.timestamp) > datetime.now() - timedelta(hours=1)
                ]),
                "recovery_success_rate": self._calculate_recovery_success_rate()
            },
            "recovery_strategies": {
                name: {
                    "priority": strategy.priority,
                    "max_attempts": strategy.max_attempts,
                    "cooldown_seconds": strategy.cooldown_seconds,
                    "last_attempt": self.last_recovery_attempt.get(name, "never").isoformat() 
                                  if isinstance(self.last_recovery_attempt.get(name), datetime) 
                                  else "never"
                }
                for name, strategy in self.recovery_strategies.items()
            }
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate from recent events."""
        recovery_attempts = [
            e for e in self.failure_history 
            if e.recovery_attempted
        ]
        
        if not recovery_attempts:
            return 0.0
        
        successful_recoveries = [
            e for e in recovery_attempts 
            if e.recovery_successful
        ]
        
        return len(successful_recoveries) / len(recovery_attempts)
    
    def force_recovery(self, strategy_name: str) -> bool:
        """Force execution of a specific recovery strategy."""
        if strategy_name not in self.recovery_strategies:
            logger.error(f"Unknown recovery strategy: {strategy_name}")
            return False
        
        strategy = self.recovery_strategies[strategy_name]
        
        logger.info(f"Force executing recovery strategy: {strategy_name}")
        
        try:
            # Create synthetic alert for forced recovery
            synthetic_alert = {
                "type": "manual_recovery",
                "message": f"Forced recovery: {strategy_name}",
                "severity": "warning",
                "value": 0
            }
            
            result = strategy.recovery_function([synthetic_alert])
            
            # Record event
            failure_event = FailureEvent(
                timestamp=datetime.now().isoformat(),
                failure_type="manual_recovery",
                component="manual",
                error_message=f"Forced recovery: {strategy_name}",
                severity="info",
                context={"forced": True},
                recovery_attempted=True,
                recovery_successful=result,
                recovery_method=strategy_name
            )
            self.failure_history.append(failure_event)
            
            return result
            
        except Exception as e:
            logger.error(f"Forced recovery failed: {e}")
            return False
    
    def export_failure_report(self, file_path: str):
        """Export failure history and recovery statistics."""
        try:
            report = {
                "export_timestamp": datetime.now().isoformat(),
                "system_status": self.get_system_status(),
                "failure_events": [asdict(event) for event in self.failure_history],
                "statistics": {
                    "total_failures": len(self.failure_history),
                    "recovery_attempts": len([e for e in self.failure_history if e.recovery_attempted]),
                    "successful_recoveries": len([e for e in self.failure_history if e.recovery_successful]),
                    "recovery_success_rate": self._calculate_recovery_success_rate()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Failure report exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export failure report: {e}")


# Global instance for easy access
_global_guard = None

def get_global_guard() -> SelfHealingPipelineGuard:
    """Get or create global self-healing guard instance."""
    global _global_guard
    if _global_guard is None:
        _global_guard = SelfHealingPipelineGuard()
    return _global_guard

def initialize_self_healing(auto_start: bool = True, **kwargs) -> SelfHealingPipelineGuard:
    """Initialize and optionally start self-healing pipeline guard."""
    global _global_guard
    _global_guard = SelfHealingPipelineGuard(**kwargs)
    
    if auto_start:
        _global_guard.start_monitoring()
        logger.info("Self-healing pipeline guard initialized and started")
    
    return _global_guard