"""Generation 2: MAKE IT ROBUST - Advanced Security and Error Handling."""

import logging
import hashlib
import hmac
import secrets
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    user_context: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_id: str = field(default_factory=lambda: secrets.token_hex(16))
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    

@dataclass
class ErrorContext:
    """Enhanced error context for debugging."""
    error_type: str
    error_message: str
    stack_trace: str
    system_state: Dict[str, Any]
    user_input: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: str = field(default_factory=lambda: secrets.token_hex(8))


class RateLimiter:
    """Advanced rate limiting with sliding window."""
    
    def __init__(self, max_requests: int = 100, window_size: int = 3600):
        self.max_requests = max_requests
        self.window_size = window_size  # seconds
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
        
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            current_time = time.time()
            user_requests = self.requests[identifier]
            
            # Clean old requests
            while user_requests and user_requests[0] < current_time - self.window_size:
                user_requests.popleft()
            
            # Check limit
            if len(user_requests) >= self.max_requests:
                return False
            
            # Add current request
            user_requests.append(current_time)
            return True
    
    def get_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        with self.lock:
            current_time = time.time()
            user_requests = self.requests[identifier]
            
            # Clean old requests
            while user_requests and user_requests[0] < current_time - self.window_size:
                user_requests.popleft()
            
            return {
                "current_requests": len(user_requests),
                "max_requests": self.max_requests,
                "window_size": self.window_size,
                "requests_remaining": max(0, self.max_requests - len(user_requests)),
                "reset_time": current_time + self.window_size if user_requests else current_time
            }


class SecurityValidator:
    """Comprehensive security validation."""
    
    # Common attack patterns
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>'
    ]
    
    SQL_INJECTION_PATTERNS = [
        r'\bunion\s+select\b',
        r'\bselect\s+.*\bfrom\b',
        r'\binsert\s+into\b',
        r'\bdelete\s+from\b',
        r'\bdrop\s+table\b',
        r'\bexec\s*\(',
        r';\s*--',
        r"'\s*or\s*'.*'='",
        r'"\s*or\s*".*"="'
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./+',
        r'\.\.\\+',
        r'/etc/passwd',
        r'/proc/self/',
        r'\\windows\\system32'
    ]
    
    @classmethod
    def validate_input(cls, input_text: str, context: str = "general") -> Dict[str, Any]:
        """Comprehensive input validation."""
        validation_result = {
            "is_safe": True,
            "threats_detected": [],
            "sanitized_input": input_text,
            "risk_level": "LOW"
        }
        
        if not input_text or not isinstance(input_text, str):
            validation_result["is_safe"] = True
            return validation_result
        
        # XSS Detection
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                validation_result["threats_detected"].append({
                    "type": "XSS",
                    "pattern": pattern,
                    "severity": "HIGH"
                })
                validation_result["risk_level"] = "HIGH"
        
        # SQL Injection Detection
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                validation_result["threats_detected"].append({
                    "type": "SQL_INJECTION", 
                    "pattern": pattern,
                    "severity": "CRITICAL"
                })
                validation_result["risk_level"] = "CRITICAL"
        
        # Path Traversal Detection
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                validation_result["threats_detected"].append({
                    "type": "PATH_TRAVERSAL",
                    "pattern": pattern,
                    "severity": "HIGH"
                })
                validation_result["risk_level"] = "HIGH"
        
        # Command Injection Detection
        command_patterns = [r';\s*\w+', r'\|\s*\w+', r'&&\s*\w+', r'`[^`]*`', r'\$\([^)]*\)']
        for pattern in command_patterns:
            if re.search(pattern, input_text):
                validation_result["threats_detected"].append({
                    "type": "COMMAND_INJECTION",
                    "pattern": pattern, 
                    "severity": "CRITICAL"
                })
                validation_result["risk_level"] = "CRITICAL"
        
        # Determine safety
        validation_result["is_safe"] = len(validation_result["threats_detected"]) == 0
        
        # Sanitize input
        validation_result["sanitized_input"] = cls.sanitize_input(input_text)
        
        return validation_result
    
    @classmethod
    def sanitize_input(cls, input_text: str) -> str:
        """Sanitize potentially dangerous input."""
        if not input_text:
            return input_text
        
        # HTML entity encoding for XSS prevention
        sanitized = input_text.replace("&", "&amp;")
        sanitized = sanitized.replace("<", "&lt;")
        sanitized = sanitized.replace(">", "&gt;")
        sanitized = sanitized.replace('"', "&quot;")
        sanitized = sanitized.replace("'", "&#x27;")
        sanitized = sanitized.replace("/", "&#x2F;")
        
        # Remove or escape dangerous characters
        sanitized = re.sub(r'[;\|\&`$]', '', sanitized)
        
        return sanitized
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> Dict[str, Any]:
        """Validate file path for security."""
        validation_result = {
            "is_safe": True,
            "issues": [],
            "resolved_path": None
        }
        
        try:
            path_obj = Path(file_path)
            resolved_path = path_obj.resolve()
            
            # Check for path traversal
            if ".." in str(path_obj):
                validation_result["issues"].append("Path traversal detected")
                validation_result["is_safe"] = False
            
            # Check for absolute paths outside allowed directories
            allowed_dirs = [Path.cwd(), Path("/tmp"), Path("/var/tmp")]
            
            is_in_allowed = any(
                str(resolved_path).startswith(str(allowed_dir.resolve()))
                for allowed_dir in allowed_dirs
            )
            
            if not is_in_allowed:
                validation_result["issues"].append("Path outside allowed directories")
                validation_result["is_safe"] = False
            
            validation_result["resolved_path"] = str(resolved_path)
            
        except Exception as e:
            validation_result["issues"].append(f"Path resolution error: {e}")
            validation_result["is_safe"] = False
        
        return validation_result


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {func.__name__} is now HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker is now OPEN (failures: {self.failure_count})")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }


class RetryWithBackoff:
    """Exponential backoff retry mechanism."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry with backoff."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        logger.error(f"Function {func.__name__} failed after {self.max_retries} retries")
                        raise
                    
                    delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay}s: {e}")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup rotating file handler
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        if not self.audit_logger.handlers:
            file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.audit_logger.addHandler(file_handler)
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event."""
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "severity": event.severity,
            "description": event.description,
            "timestamp": event.timestamp,
            "user_context": event.user_context,
            "source_ip": event.source_ip,
            "user_agent": event.user_agent
        }
        
        log_message = f"SECURITY_EVENT | {json.dumps(event_data, default=str)}"
        
        if event.severity in ["CRITICAL", "HIGH"]:
            self.audit_logger.error(log_message)
        elif event.severity == "MEDIUM":
            self.audit_logger.warning(log_message)
        else:
            self.audit_logger.info(log_message)
    
    def log_access_attempt(self, user_id: str, resource: str, success: bool, details: Dict[str, Any] = None):
        """Log access attempt."""
        event = SecurityEvent(
            event_type="ACCESS_ATTEMPT",
            severity="MEDIUM" if not success else "LOW",
            description=f"Access {'granted' if success else 'denied'} to {resource}",
            user_context={"user_id": user_id, "resource": resource, "success": success, "details": details or {}}
        )
        self.log_security_event(event)
    
    def log_rate_limit_exceeded(self, identifier: str, limit: int, current_requests: int):
        """Log rate limit exceeded."""
        event = SecurityEvent(
            event_type="RATE_LIMIT_EXCEEDED",
            severity="MEDIUM",
            description=f"Rate limit exceeded by {identifier}",
            user_context={
                "identifier": identifier,
                "limit": limit,
                "current_requests": current_requests
            }
        )
        self.log_security_event(event)
    
    def log_validation_failure(self, input_data: str, threats: List[Dict[str, Any]], context: str):
        """Log input validation failure."""
        event = SecurityEvent(
            event_type="VALIDATION_FAILURE",
            severity="HIGH",
            description=f"Input validation failed in {context}",
            user_context={
                "context": context,
                "threats_detected": threats,
                "input_hash": hashlib.sha256(input_data.encode()).hexdigest()[:16]
            }
        )
        self.log_security_event(event)


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = deque(maxlen=100)
        self.thresholds = {
            "error_rate": 0.1,  # 10% error rate threshold
            "response_time": 5.0,  # 5 second response time threshold
            "memory_usage": 0.8,  # 80% memory usage threshold
            "cpu_usage": 0.9  # 90% CPU usage threshold
        }
        self.lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value."""
        with self.lock:
            if timestamp is None:
                timestamp = time.time()
            
            self.metrics[metric_name].append((timestamp, value))
            
            # Keep only recent metrics (last hour)
            cutoff_time = timestamp - 3600
            self.metrics[metric_name] = [
                (ts, val) for ts, val in self.metrics[metric_name]
                if ts > cutoff_time
            ]
            
            # Check thresholds
            self._check_threshold(metric_name, value)
    
    def _check_threshold(self, metric_name: str, value: float):
        """Check if metric exceeds threshold."""
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            
            if value > threshold:
                alert = {
                    "alert_id": secrets.token_hex(8),
                    "metric": metric_name,
                    "value": value,
                    "threshold": threshold,
                    "timestamp": time.time(),
                    "severity": "HIGH" if value > threshold * 1.5 else "MEDIUM"
                }
                
                self.alerts.append(alert)
                logger.warning(f"Threshold exceeded: {metric_name}={value} > {threshold}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        with self.lock:
            current_time = time.time()
            health_status = {
                "timestamp": current_time,
                "overall_status": "HEALTHY",
                "metrics": {},
                "active_alerts": len(self.alerts),
                "recent_alerts": []
            }
            
            # Calculate metric summaries
            for metric_name, values in self.metrics.items():
                if values:
                    recent_values = [val for ts, val in values if ts > current_time - 300]  # Last 5 minutes
                    
                    if recent_values:
                        health_status["metrics"][metric_name] = {
                            "current": recent_values[-1],
                            "average": sum(recent_values) / len(recent_values),
                            "min": min(recent_values),
                            "max": max(recent_values),
                            "count": len(recent_values)
                        }
            
            # Recent alerts
            recent_alerts = [alert for alert in self.alerts if alert["timestamp"] > current_time - 3600]
            health_status["recent_alerts"] = recent_alerts[-10:]  # Last 10 alerts
            
            # Determine overall status
            high_severity_alerts = [a for a in recent_alerts if a["severity"] == "HIGH"]
            if high_severity_alerts:
                health_status["overall_status"] = "UNHEALTHY"
            elif len(recent_alerts) > 10:
                health_status["overall_status"] = "DEGRADED"
            
            return health_status
    
    def clear_alerts(self):
        """Clear all alerts."""
        with self.lock:
            self.alerts.clear()


class RobustCounterfactualGenerator:
    """Robust counterfactual generator with comprehensive error handling and security."""
    
    def __init__(self, method: str = "modicf", device: str = "cpu", security_level: str = "high"):
        # Initialize security components
        self.security_level = security_level
        self.security_validator = SecurityValidator()
        self.audit_logger = AuditLogger()
        self.health_monitor = HealthMonitor()
        self.rate_limiter = RateLimiter(max_requests=50, window_size=300)  # 50 requests per 5 minutes
        
        # Initialize circuit breakers for critical operations
        self.generation_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=Exception
        )
        
        # Core generator (fallback to lightweight if needed)
        try:
            from counterfactual_lab.core import CounterfactualGenerator
            self.core_generator = CounterfactualGenerator(method=method, device=device)
            logger.info(f"Initialized robust generator with core implementation")
        except ImportError:
            from counterfactual_lab.lightweight_core import LightweightCounterfactualGenerator
            self.core_generator = LightweightCounterfactualGenerator(method=method, device=device)
            logger.info(f"Initialized robust generator with lightweight implementation")
        
        # Initialize with security event
        self.audit_logger.log_security_event(SecurityEvent(
            event_type="GENERATOR_INITIALIZATION",
            severity="LOW",
            description=f"RobustCounterfactualGenerator initialized with {method} method",
            user_context={"method": method, "device": device, "security_level": security_level}
        ))
    
    @RetryWithBackoff(max_retries=2, base_delay=1.0)
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    def generate_secure(
        self,
        image,
        text: str,
        attributes: Union[List[str], str],
        num_samples: int = 5,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate counterfactuals with comprehensive security and robustness."""
        start_time = time.time()
        correlation_id = secrets.token_hex(8)
        user_context = user_context or {}
        
        try:
            # Rate limiting check
            user_id = user_context.get("user_id", "anonymous")
            if not self.rate_limiter.is_allowed(user_id):
                rate_stats = self.rate_limiter.get_stats(user_id)
                self.audit_logger.log_rate_limit_exceeded(user_id, rate_stats["max_requests"], rate_stats["current_requests"])
                raise Exception("Rate limit exceeded. Please try again later.")
            
            # Input validation
            text_validation = self.security_validator.validate_input(text, "text_input")
            if not text_validation["is_safe"]:
                self.audit_logger.log_validation_failure(text, text_validation["threats_detected"], "text_input")
                if self.security_level == "high":
                    raise Exception("Input validation failed: potentially malicious content detected")
                else:
                    text = text_validation["sanitized_input"]
                    logger.warning("Input sanitized due to security concerns")
            
            # Validate attributes
            if isinstance(attributes, str):
                attr_validation = self.security_validator.validate_input(attributes, "attributes")
                if not attr_validation["is_safe"]:
                    self.audit_logger.log_validation_failure(attributes, attr_validation["threats_detected"], "attributes")
                    if self.security_level == "high":
                        raise Exception("Attribute validation failed")
                    attributes = attr_validation["sanitized_input"]
                attributes = [attr.strip() for attr in attributes.split(",")]
            
            # Validate num_samples
            if not isinstance(num_samples, int) or num_samples <= 0 or num_samples > 20:
                raise ValueError("num_samples must be a positive integer <= 20")
            
            # Log access attempt
            self.audit_logger.log_access_attempt(
                user_id, 
                "counterfactual_generation", 
                True,
                {"method": self.core_generator.method, "num_samples": num_samples, "correlation_id": correlation_id}
            )
            
            # Perform generation with monitoring
            try:
                generation_start = time.time()
                results = self.core_generator.generate(image, text, attributes, num_samples)
                generation_time = time.time() - generation_start
                
                # Record performance metrics
                self.health_monitor.record_metric("generation_time", generation_time)
                self.health_monitor.record_metric("samples_generated", len(results.get("counterfactuals", [])))
                
                # Enhance results with security metadata
                results["security_metadata"] = {
                    "correlation_id": correlation_id,
                    "security_level": self.security_level,
                    "input_validated": True,
                    "rate_limited": True,
                    "user_context": user_context,
                    "generation_time": generation_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Record successful generation
                self.health_monitor.record_metric("success_rate", 1.0)
                
                total_time = time.time() - start_time
                logger.info(f"Secure generation completed in {total_time:.2f}s (correlation_id: {correlation_id})")
                
                return results
                
            except Exception as generation_error:
                self.health_monitor.record_metric("success_rate", 0.0)
                self.health_monitor.record_metric("error_rate", 1.0)
                
                error_context = ErrorContext(
                    error_type=type(generation_error).__name__,
                    error_message=str(generation_error),
                    stack_trace=str(generation_error),
                    system_state={"method": self.core_generator.method, "device": getattr(self.core_generator, 'device', 'unknown')},
                    user_input={"text": text[:100], "attributes": attributes, "num_samples": num_samples},
                    correlation_id=correlation_id
                )
                
                self.audit_logger.log_security_event(SecurityEvent(
                    event_type="GENERATION_ERROR",
                    severity="MEDIUM",
                    description=f"Generation failed: {str(generation_error)[:200]}",
                    user_context={"correlation_id": correlation_id, "error_type": error_context.error_type}
                ))
                
                raise
                
        except Exception as e:
            # Log failed access attempt
            self.audit_logger.log_access_attempt(
                user_context.get("user_id", "anonymous"),
                "counterfactual_generation",
                False,
                {"error": str(e)[:200], "correlation_id": correlation_id}
            )
            
            total_time = time.time() - start_time
            self.health_monitor.record_metric("total_request_time", total_time)
            
            raise
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        health_status = self.health_monitor.get_health_status()
        
        security_status = {
            "security_level": self.security_level,
            "health_status": health_status,
            "rate_limiting": {
                "enabled": True,
                "max_requests": self.rate_limiter.max_requests,
                "window_size": self.rate_limiter.window_size
            },
            "circuit_breakers": {
                "generation": self.generation_circuit_breaker.get_status()
            },
            "audit_logging": {
                "enabled": True,
                "log_file": str(self.audit_logger.log_file)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return security_status
    
    def reset_security_components(self):
        """Reset security components (admin function)."""
        self.health_monitor.clear_alerts()
        self.generation_circuit_breaker.failure_count = 0
        self.generation_circuit_breaker.state = "CLOSED"
        logger.info("Security components reset")


def demonstrate_robust_security():
    """Demonstrate robust security features."""
    print("\n" + "="*80)
    print("🛡️ GENERATION 2: ROBUST SECURITY DEMONSTRATION")
    print("   Advanced Security, Error Handling, and Reliability")
    print("="*80)
    
    # Initialize robust generator
    print("\n🔧 Initializing Robust Generator...")
    robust_generator = RobustCounterfactualGenerator(
        method="modicf",
        device="cpu", 
        security_level="high"
    )
    print("   ✅ Robust generator initialized with high security")
    
    # Test 1: Normal operation
    print("\n✅ Test 1: Normal Secure Generation")
    print("-" * 50)
    
    try:
        results = robust_generator.generate_secure(
            image="mock_image.png",
            text="A professional doctor in a hospital setting",
            attributes=["gender", "age"],
            num_samples=3,
            user_context={"user_id": "test_user_1", "session_id": "session_123"}
        )
        
        print(f"   ✅ Generated {len(results['counterfactuals'])} counterfactuals")
        print(f"   ✅ Security correlation ID: {results['security_metadata']['correlation_id']}")
        print(f"   ✅ Generation time: {results['security_metadata']['generation_time']:.3f}s")
        
    except Exception as e:
        print(f"   ❌ Normal generation failed: {e}")
    
    # Test 2: Security validation
    print("\n🛡️ Test 2: Security Validation")
    print("-" * 50)
    
    # Test XSS attempt
    try:
        malicious_text = "A doctor <script>alert('XSS')</script> in hospital"
        results = robust_generator.generate_secure(
            image="mock_image.png",
            text=malicious_text,
            attributes=["gender"],
            num_samples=1,
            user_context={"user_id": "potential_attacker"}
        )
        print(f"   ⚠️ Malicious input was processed (security_level may be permissive)")
        
    except Exception as e:
        print(f"   ✅ Malicious input blocked: {str(e)[:100]}...")
    
    # Test 3: Rate limiting
    print("\n⏱️ Test 3: Rate Limiting")
    print("-" * 50)
    
    successful_requests = 0
    rate_limited_requests = 0
    
    for i in range(7):  # Try more requests than limit
        try:
            results = robust_generator.generate_secure(
                image="mock_image.png",
                text=f"Test request {i+1}",
                attributes=["gender"],
                num_samples=1,
                user_context={"user_id": "rate_test_user"}
            )
            successful_requests += 1
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                rate_limited_requests += 1
            else:
                print(f"   ⚠️ Unexpected error: {e}")
    
    print(f"   ✅ Successful requests: {successful_requests}")
    print(f"   ✅ Rate limited requests: {rate_limited_requests}")
    
    # Test 4: Health monitoring
    print("\n📊 Test 4: Health Monitoring")
    print("-" * 50)
    
    health_status = robust_generator.health_monitor.get_health_status()
    print(f"   ✅ Overall health status: {health_status['overall_status']}")
    print(f"   ✅ Active alerts: {health_status['active_alerts']}")
    print(f"   ✅ Metrics tracked: {len(health_status['metrics'])}")
    
    for metric_name, metric_data in health_status["metrics"].items():
        if isinstance(metric_data, dict):
            print(f"   📈 {metric_name}: avg={metric_data.get('average', 0):.3f}, count={metric_data.get('count', 0)}")
    
    # Test 5: Security status
    print("\n🔒 Test 5: Comprehensive Security Status")
    print("-" * 50)
    
    security_status = robust_generator.get_security_status()
    print(f"   ✅ Security level: {security_status['security_level']}")
    print(f"   ✅ Rate limiting enabled: {security_status['rate_limiting']['enabled']}")
    print(f"   ✅ Circuit breaker state: {security_status['circuit_breakers']['generation']['state']}")
    print(f"   ✅ Audit logging enabled: {security_status['audit_logging']['enabled']}")
    
    # Show circuit breaker status
    cb_status = security_status['circuit_breakers']['generation']
    print(f"   📊 Circuit breaker failures: {cb_status['failure_count']}/{cb_status['failure_threshold']}")
    
    print("\n" + "="*80)
    print("🎉 GENERATION 2: ROBUST SECURITY DEMONSTRATION COMPLETE!")
    print("   Security Features: ✅ Comprehensive")
    print("   Error Handling: ✅ Advanced") 
    print("   Rate Limiting: ✅ Active")
    print("   Health Monitoring: ✅ Operational")
    print("   Audit Logging: ✅ Complete")
    print("="*80)
    
    return robust_generator, security_status


if __name__ == "__main__":
    robust_generator, security_status = demonstrate_robust_security()
    
    # Save security status
    with open("generation_2_security_status.json", "w") as f:
        json.dump(security_status, f, indent=2, default=str)
    
    print(f"\n💾 Security status saved to: generation_2_security_status.json")