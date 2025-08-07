"""System monitoring and health checks for counterfactual generation."""

import psutil
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_available: bool
    gpu_memory_usage: Optional[float]
    cache_status: str
    storage_status: str
    timestamp: str
    issues: List[str]


class HealthMonitor:
    """Monitors system health and performance metrics."""
    
    def __init__(self, check_interval: int = 60):
        """Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.last_check = None
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 100
        
    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check."""
        timestamp = datetime.now().isoformat()
        issues = []
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 90:
            issues.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        if memory_usage > 85:
            issues.append(f"High memory usage: {memory_usage:.1f}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        if disk_usage > 90:
            issues.append(f"High disk usage: {disk_usage:.1f}%")
        
        # GPU availability and memory
        gpu_available = False
        gpu_memory_usage = None
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                gpu_memory_usage = (gpu_used / gpu_memory) * 100
                
                if gpu_memory_usage > 90:
                    issues.append(f"High GPU memory usage: {gpu_memory_usage:.1f}%")
        except ImportError:
            pass
        
        # Cache and storage status
        cache_status = "healthy"
        storage_status = "healthy"
        
        # Overall status
        if not issues:
            overall_status = "healthy"
        elif len(issues) <= 2:
            overall_status = "warning"
        else:
            overall_status = "critical"
        
        health = SystemHealth(
            overall_status=overall_status,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            gpu_available=gpu_available,
            gpu_memory_usage=gpu_memory_usage,
            cache_status=cache_status,
            storage_status=storage_status,
            timestamp=timestamp,
            issues=issues
        )
        
        # Store in history
        self.health_history.append(health)
        if len(self.health_history) > self.max_history_size:
            self.health_history.pop(0)
        
        self.last_check = datetime.now()
        
        if issues:
            logger.warning(f"System health issues detected: {', '.join(issues)}")
        else:
            logger.debug("System health check passed")
        
        return health
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for the last N hours."""
        if not self.health_history:
            return {"error": "No health data available"}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            h for h in self.health_history
            if datetime.fromisoformat(h.timestamp) > cutoff_time
        ]
        
        if not recent_checks:
            return {"error": f"No health data in last {hours} hours"}
        
        # Calculate averages and trends
        avg_cpu = sum(h.cpu_usage for h in recent_checks) / len(recent_checks)
        avg_memory = sum(h.memory_usage for h in recent_checks) / len(recent_checks)
        avg_disk = sum(h.disk_usage for h in recent_checks) / len(recent_checks)
        
        gpu_usage_values = [h.gpu_memory_usage for h in recent_checks if h.gpu_memory_usage is not None]
        avg_gpu = sum(gpu_usage_values) / len(gpu_usage_values) if gpu_usage_values else None
        
        # Count status occurrences
        status_counts = {}
        for h in recent_checks:
            status_counts[h.overall_status] = status_counts.get(h.overall_status, 0) + 1
        
        # Collect all unique issues
        all_issues = set()
        for h in recent_checks:
            all_issues.update(h.issues)
        
        return {
            "period_hours": hours,
            "total_checks": len(recent_checks),
            "latest_status": recent_checks[-1].overall_status,
            "averages": {
                "cpu_usage": round(avg_cpu, 1),
                "memory_usage": round(avg_memory, 1),
                "disk_usage": round(avg_disk, 1),
                "gpu_memory_usage": round(avg_gpu, 1) if avg_gpu else None
            },
            "status_distribution": status_counts,
            "common_issues": list(all_issues),
            "last_check": recent_checks[-1].timestamp
        }
    
    def export_health_data(self, file_path: str):
        """Export health history to JSON file."""
        try:
            health_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_records": len(self.health_history),
                "health_history": [
                    {
                        "timestamp": h.timestamp,
                        "overall_status": h.overall_status,
                        "cpu_usage": h.cpu_usage,
                        "memory_usage": h.memory_usage,
                        "disk_usage": h.disk_usage,
                        "gpu_available": h.gpu_available,
                        "gpu_memory_usage": h.gpu_memory_usage,
                        "issues": h.issues
                    }
                    for h in self.health_history
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(health_data, f, indent=2)
            
            logger.info(f"Health data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export health data: {e}")


class PerformanceProfiler:
    """Profiles performance of generation operations."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        
    def start_operation(self, operation_name: str) -> float:
        """Start timing an operation."""
        return time.time()
    
    def end_operation(self, operation_name: str, start_time: float):
        """End timing an operation."""
        duration = time.time() - start_time
        
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        # Keep only recent measurements
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name].pop(0)
        
        logger.debug(f"{operation_name} completed in {duration:.3f}s")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    "count": self.operation_counts[operation],
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "recent_samples": len(times)
                }
        
        return stats
    
    def get_performance_report(self) -> str:
        """Generate human-readable performance report."""
        stats = self.get_performance_stats()
        
        if not stats:
            return "No performance data available"
        
        report_lines = ["Performance Report", "=" * 50]
        
        for operation, data in stats.items():
            report_lines.extend([
                f"\n{operation}:",
                f"  Total operations: {data['count']}",
                f"  Average time: {data['avg_time']:.3f}s",
                f"  Min time: {data['min_time']:.3f}s",
                f"  Max time: {data['max_time']:.3f}s",
                f"  Recent samples: {data['recent_samples']}"
            ])
        
        return "\n".join(report_lines)


class AlertManager:
    """Manages alerts and notifications for system issues."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "gpu_memory_usage": 90.0,
            "generation_time": 60.0  # seconds
        }
        
    def check_thresholds(self, health: SystemHealth, generation_time: Optional[float] = None):
        """Check if any thresholds are exceeded."""
        alerts_triggered = []
        
        # System resource alerts
        if health.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts_triggered.append({
                "type": "cpu_high",
                "message": f"CPU usage high: {health.cpu_usage:.1f}%",
                "severity": "warning" if health.cpu_usage < 95 else "critical",
                "value": health.cpu_usage,
                "threshold": self.alert_thresholds["cpu_usage"]
            })
        
        if health.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts_triggered.append({
                "type": "memory_high",
                "message": f"Memory usage high: {health.memory_usage:.1f}%",
                "severity": "warning" if health.memory_usage < 90 else "critical",
                "value": health.memory_usage,
                "threshold": self.alert_thresholds["memory_usage"]
            })
        
        if health.disk_usage > self.alert_thresholds["disk_usage"]:
            alerts_triggered.append({
                "type": "disk_high",
                "message": f"Disk usage high: {health.disk_usage:.1f}%",
                "severity": "warning" if health.disk_usage < 95 else "critical",
                "value": health.disk_usage,
                "threshold": self.alert_thresholds["disk_usage"]
            })
        
        if (health.gpu_memory_usage and 
            health.gpu_memory_usage > self.alert_thresholds["gpu_memory_usage"]):
            alerts_triggered.append({
                "type": "gpu_memory_high",
                "message": f"GPU memory usage high: {health.gpu_memory_usage:.1f}%",
                "severity": "warning" if health.gpu_memory_usage < 95 else "critical",
                "value": health.gpu_memory_usage,
                "threshold": self.alert_thresholds["gpu_memory_usage"]
            })
        
        # Performance alerts
        if generation_time and generation_time > self.alert_thresholds["generation_time"]:
            alerts_triggered.append({
                "type": "generation_slow",
                "message": f"Generation time slow: {generation_time:.1f}s",
                "severity": "warning" if generation_time < 120 else "critical",
                "value": generation_time,
                "threshold": self.alert_thresholds["generation_time"]
            })
        
        # Store alerts
        for alert in alerts_triggered:
            alert["timestamp"] = datetime.now().isoformat()
            self.alerts.append(alert)
            
            # Log alert
            if alert["severity"] == "critical":
                logger.error(f"CRITICAL ALERT: {alert['message']}")
            else:
                logger.warning(f"ALERT: {alert['message']}")
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
        
        return alerts_triggered
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.alerts if alert["severity"] == severity]
        return self.alerts
    
    def clear_alerts(self, alert_type: Optional[str] = None):
        """Clear alerts, optionally by type."""
        if alert_type:
            self.alerts = [alert for alert in self.alerts if alert["type"] != alert_type]
        else:
            self.alerts.clear()
        
        logger.info(f"Cleared alerts" + (f" of type {alert_type}" if alert_type else ""))


class SystemDiagnostics:
    """Comprehensive system diagnostics."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.performance_profiler = PerformanceProfiler()
        self.alert_manager = AlertManager()
    
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run complete system diagnostics."""
        logger.info("Running full system diagnostics...")
        
        # Health check
        health = self.health_monitor.check_system_health()
        
        # Performance stats
        performance_stats = self.performance_profiler.get_performance_stats()
        
        # Check alerts
        alerts = self.alert_manager.check_thresholds(health)
        
        # Environment info
        env_info = self._get_environment_info()
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "health": {
                "overall_status": health.overall_status,
                "cpu_usage": health.cpu_usage,
                "memory_usage": health.memory_usage,
                "disk_usage": health.disk_usage,
                "gpu_available": health.gpu_available,
                "gpu_memory_usage": health.gpu_memory_usage,
                "issues": health.issues
            },
            "performance": performance_stats,
            "alerts": {
                "total_alerts": len(self.alert_manager.alerts),
                "recent_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
                "active_alerts": alerts
            },
            "environment": env_info
        }
        
        logger.info("System diagnostics completed")
        return diagnostics
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        env_info = {
            "python_version": None,
            "pytorch_version": None,
            "cuda_version": None,
            "platform": None,
            "total_memory": None,
            "available_disk_space": None
        }
        
        try:
            import sys
            env_info["python_version"] = sys.version
            
            import platform
            env_info["platform"] = platform.platform()
            
            memory = psutil.virtual_memory()
            env_info["total_memory"] = f"{memory.total / (1024**3):.1f} GB"
            
            disk = psutil.disk_usage('/')
            env_info["available_disk_space"] = f"{disk.free / (1024**3):.1f} GB"
            
            try:
                import torch
                env_info["pytorch_version"] = torch.__version__
                if torch.cuda.is_available():
                    env_info["cuda_version"] = torch.version.cuda
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to get environment info: {e}")
        
        return env_info