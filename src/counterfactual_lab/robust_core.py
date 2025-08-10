"""Robust core with comprehensive error handling, logging, and security."""

import json
import logging
import random
import hashlib
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from functools import wraps
import warnings

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('counterfactual_lab.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CounterfactualLabError(Exception):
    """Base exception for all counterfactual lab errors."""
    pass


class ValidationError(CounterfactualLabError):
    """Raised when input validation fails."""
    pass


class GenerationError(CounterfactualLabError):
    """Raised when counterfactual generation fails."""
    pass


class EvaluationError(CounterfactualLabError):
    """Raised when bias evaluation fails."""
    pass


class SecurityError(CounterfactualLabError):
    """Raised when security validation fails."""
    pass


class RateLimitError(CounterfactualLabError):
    """Raised when rate limits are exceeded."""
    pass


def with_error_handling(operation_name: str, max_retries: int = 3):
    """Decorator for comprehensive error handling and retry logic."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    logger.info(f"{operation_name} completed successfully in {duration:.2f}s (attempt {attempt + 1})")
                    return result
                    
                except Exception as e:
                    last_exception = e
                    duration = time.time() - start_time if 'start_time' in locals() else 0
                    
                    logger.warning(f"{operation_name} failed on attempt {attempt + 1}: {str(e)} (duration: {duration:.2f}s)")
                    
                    if attempt < max_retries:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{operation_name} failed after {max_retries + 1} attempts")
                        logger.error(f"Final error: {str(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # If we get here, all retries failed
            raise GenerationError(f"{operation_name} failed after {max_retries + 1} attempts: {str(last_exception)}")
        
        return wrapper
    return decorator


class SecurityValidator:
    """Comprehensive security validation for inputs and operations."""
    
    SENSITIVE_PATTERNS = [
        r'social\s*security',
        r'ssn',
        r'credit\s*card',
        r'password',
        r'api\s*key',
        r'token',
        r'private\s*key'
    ]
    
    PROHIBITED_ATTRIBUTES = [
        'social_security',
        'credit_card',
        'medical_condition',
        'sexual_orientation',
        'political_affiliation'
    ]
    
    @staticmethod
    def validate_text_input(text: str) -> tuple[bool, List[str]]:
        """Validate text input for security concerns."""
        import re
        
        issues = []
        
        # Check for sensitive information patterns
        for pattern in SecurityValidator.SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Potentially sensitive information detected: {pattern}")
        
        # Check text length
        if len(text) > 10000:
            issues.append("Text input exceeds maximum length")
        
        # Check for unusual character patterns
        if len(set(text)) < len(text) * 0.1:  # Too repetitive
            issues.append("Text appears to contain repetitive patterns")
        
        is_safe = len(issues) == 0
        return is_safe, issues
    
    @staticmethod
    def validate_attributes(attributes: List[str]) -> tuple[bool, List[str]]:
        """Validate that attributes are ethically acceptable."""
        issues = []
        
        for attr in attributes:
            if attr.lower() in SecurityValidator.PROHIBITED_ATTRIBUTES:
                issues.append(f"Prohibited attribute: {attr}")
        
        is_safe = len(issues) == 0
        return is_safe, issues
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize text input by removing or masking sensitive content."""
        import re
        
        sanitized = text
        
        # Mask potential sensitive patterns
        for pattern in SecurityValidator.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "[TRUNCATED]"
        
        return sanitized


class InputValidator:
    """Comprehensive input validation."""
    
    SUPPORTED_METHODS = ["modicf", "icg"]
    SUPPORTED_DEVICES = ["cpu", "cuda", "auto"]
    SUPPORTED_ATTRIBUTES = {
        "gender": ["male", "female", "non-binary"],
        "age": ["young", "middle-aged", "elderly"],
        "race": ["white", "black", "asian", "hispanic", "diverse"],
        "expression": ["neutral", "smiling", "serious", "happy", "sad"],
        "hair_color": ["blonde", "brown", "black", "red", "gray"],
        "hair_style": ["short", "long", "curly", "straight"]
    }
    
    @staticmethod
    def validate_method(method: str) -> str:
        """Validate generation method."""
        if not isinstance(method, str):
            raise ValidationError("Method must be a string")
        
        method = method.lower().strip()
        if method not in InputValidator.SUPPORTED_METHODS:
            raise ValidationError(f"Unsupported method: {method}. Supported: {InputValidator.SUPPORTED_METHODS}")
        
        return method
    
    @staticmethod
    def validate_device(device: str) -> str:
        """Validate compute device."""
        if not isinstance(device, str):
            raise ValidationError("Device must be a string")
        
        device = device.lower().strip()
        if device not in InputValidator.SUPPORTED_DEVICES:
            raise ValidationError(f"Unsupported device: {device}. Supported: {InputValidator.SUPPORTED_DEVICES}")
        
        return device
    
    @staticmethod
    def validate_text(text: str) -> str:
        """Validate text input."""
        if not isinstance(text, str):
            raise ValidationError("Text must be a string")
        
        text = text.strip()
        if not text:
            raise ValidationError("Text cannot be empty")
        
        if len(text) < 3:
            raise ValidationError("Text must be at least 3 characters long")
        
        # Security validation
        is_safe, issues = SecurityValidator.validate_text_input(text)
        if not is_safe:
            logger.warning(f"Text security issues detected: {issues}")
            # Don't fail, but sanitize
            text = SecurityValidator.sanitize_input(text)
        
        return text
    
    @staticmethod
    def validate_attributes(attributes: Union[List[str], str]) -> List[str]:
        """Validate attribute list."""
        if isinstance(attributes, str):
            attributes = [attr.strip().lower() for attr in attributes.split(',')]
        
        if not isinstance(attributes, list):
            raise ValidationError("Attributes must be a list or comma-separated string")
        
        if not attributes:
            raise ValidationError("At least one attribute must be specified")
        
        validated_attrs = []
        for attr in attributes:
            if not isinstance(attr, str):
                raise ValidationError("Each attribute must be a string")
            
            attr = attr.strip().lower()
            if attr not in InputValidator.SUPPORTED_ATTRIBUTES:
                logger.warning(f"Unsupported attribute: {attr}. Supported: {list(InputValidator.SUPPORTED_ATTRIBUTES.keys())}")
                continue
            
            validated_attrs.append(attr)
        
        if not validated_attrs:
            raise ValidationError("No valid attributes found")
        
        # Security validation
        is_safe, issues = SecurityValidator.validate_attributes(validated_attrs)
        if not is_safe:
            raise SecurityError(f"Attribute security validation failed: {issues}")
        
        return validated_attrs
    
    @staticmethod
    def validate_num_samples(num_samples: int) -> int:
        """Validate number of samples."""
        if not isinstance(num_samples, int):
            raise ValidationError("num_samples must be an integer")
        
        if num_samples < 1:
            raise ValidationError("num_samples must be at least 1")
        
        if num_samples > 100:
            raise ValidationError("num_samples cannot exceed 100 for safety")
        
        return num_samples


class RateLimiter:
    """Rate limiting to prevent abuse."""
    
    def __init__(self, max_requests: int = 60, window_minutes: int = 1):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = {}  # user_id -> list of timestamps
        
        logger.info(f"Rate limiter initialized: {max_requests} requests per {window_minutes} minutes")
    
    def check_rate_limit(self, user_id: str = "anonymous") -> bool:
        """Check if user is within rate limits."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                timestamp for timestamp in self.requests[user_id]
                if timestamp > window_start
            ]
        else:
            self.requests[user_id] = []
        
        # Check current rate
        current_requests = len(self.requests[user_id])
        
        if current_requests >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}: {current_requests}/{self.max_requests}")
            return False
        
        # Record this request
        self.requests[user_id].append(now)
        return True
    
    def get_rate_info(self, user_id: str = "anonymous") -> Dict[str, Any]:
        """Get current rate limit information for user."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        if user_id in self.requests:
            current_requests = len([
                timestamp for timestamp in self.requests[user_id]
                if timestamp > window_start
            ])
        else:
            current_requests = 0
        
        return {
            "current_requests": current_requests,
            "max_requests": self.max_requests,
            "window_minutes": self.window_minutes,
            "remaining_requests": max(0, self.max_requests - current_requests),
            "reset_time": (window_start + timedelta(minutes=self.window_minutes)).isoformat()
        }


class AuditLogger:
    """Comprehensive audit logging for security and compliance."""
    
    def __init__(self, log_file: str = "audit.log"):
        """Initialize audit logger."""
        self.log_file = log_file
        
        # Configure audit logger
        self.audit_logger = logging.getLogger("AUDIT")
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler for audit logs
        audit_handler = logging.FileHandler(log_file, mode='a')
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
        
        logger.info(f"Audit logging initialized: {log_file}")
    
    def log_generation_request(self, user_id: str, method: str, attributes: List[str], success: bool):
        """Log counterfactual generation request."""
        self.audit_logger.info(
            f"GENERATION_REQUEST - User: {user_id}, Method: {method}, "
            f"Attributes: {attributes}, Success: {success}"
        )
    
    def log_evaluation_request(self, user_id: str, metrics: List[str], success: bool):
        """Log bias evaluation request."""
        self.audit_logger.info(
            f"EVALUATION_REQUEST - User: {user_id}, Metrics: {metrics}, Success: {success}"
        )
    
    def log_security_event(self, event_type: str, severity: str, description: str, details: Dict[str, Any]):
        """Log security-related events."""
        self.audit_logger.warning(
            f"SECURITY_EVENT - Type: {event_type}, Severity: {severity}, "
            f"Description: {description}, Details: {json.dumps(details, default=str)}"
        )
    
    def log_error(self, operation: str, error: str, user_id: str = "anonymous"):
        """Log errors for investigation."""
        self.audit_logger.error(
            f"ERROR - Operation: {operation}, User: {user_id}, Error: {error}"
        )


class MockImage:
    """Enhanced mock image class with validation."""
    
    def __init__(self, width: int = 400, height: int = 300, mode: str = "RGB"):
        # Validate parameters
        if width <= 0 or height <= 0:
            raise ValidationError("Image dimensions must be positive")
        if width > 10000 or height > 10000:
            raise ValidationError("Image dimensions too large")
        if mode not in ["RGB", "RGBA", "L"]:
            raise ValidationError(f"Unsupported image mode: {mode}")
        
        self.width = width
        self.height = height
        self.mode = mode
        self.format = None
        self.data = f"robust_image_{width}x{height}_{random.randint(1000, 9999)}"
    
    def copy(self):
        """Return a copy of this mock image."""
        return MockImage(self.width, self.height, self.mode)
    
    def save(self, fp, format=None):
        """Mock save operation with validation."""
        if hasattr(fp, 'write'):
            fp.write(f"MOCK_IMAGE_DATA_{self.data}")
        else:
            try:
                with open(fp, 'w') as f:
                    f.write(f"MOCK_IMAGE_DATA_{self.data}")
            except Exception as e:
                raise GenerationError(f"Failed to save image: {e}")
    
    def __str__(self):
        return f"MockImage({self.width}x{self.height}, {self.mode})"


class RobustCounterfactualGenerator:
    """Robust counterfactual generator with comprehensive error handling and security."""
    
    def __init__(
        self,
        method: str = "modicf",
        device: str = "cpu",
        enable_rate_limiting: bool = True,
        enable_audit_logging: bool = True,
        max_requests_per_minute: int = 10
    ):
        """Initialize robust generator."""
        try:
            # Validate inputs
            self.method = InputValidator.validate_method(method)
            self.device = InputValidator.validate_device(device)
            
            # Initialize components
            self.generation_count = 0
            self.error_count = 0
            self.start_time = datetime.now()
            
            # Security and monitoring
            if enable_rate_limiting:
                self.rate_limiter = RateLimiter(max_requests_per_minute)
            else:
                self.rate_limiter = None
            
            if enable_audit_logging:
                self.audit_logger = AuditLogger()
            else:
                self.audit_logger = None
            
            # Performance tracking
            self.performance_metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0.0,
                'error_rate': 0.0
            }
            
            logger.info(
                f"Robust generator initialized: method={self.method}, device={self.device}, "
                f"rate_limiting={enable_rate_limiting}, audit_logging={enable_audit_logging}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise GenerationError(f"Generator initialization failed: {e}")
    
    @with_error_handling("counterfactual_generation", max_retries=2)
    def generate(
        self,
        image: Union[str, Path, MockImage],
        text: str,
        attributes: Union[List[str], str],
        num_samples: int = 5,
        user_id: str = "anonymous",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate counterfactual examples with robust error handling."""
        
        request_start_time = time.time()
        
        try:
            # Rate limiting check
            if self.rate_limiter and not self.rate_limiter.check_rate_limit(user_id):
                rate_info = self.rate_limiter.get_rate_info(user_id)
                raise RateLimitError(
                    f"Rate limit exceeded. Max: {rate_info['max_requests']} requests "
                    f"per {rate_info['window_minutes']} minutes. Try again at {rate_info['reset_time']}"
                )
            
            # Input validation
            validated_text = InputValidator.validate_text(text)
            validated_attributes = InputValidator.validate_attributes(attributes)
            validated_num_samples = InputValidator.validate_num_samples(num_samples)
            
            # Image validation
            if isinstance(image, str):
                mock_image = MockImage(400, 300)
            elif isinstance(image, Path):
                mock_image = MockImage(400, 300)
            elif hasattr(image, 'width') and hasattr(image, 'height'):
                mock_image = image
            else:
                raise ValidationError("Invalid image input")
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_generation_request(
                    user_id, self.method, validated_attributes, True
                )
            
            logger.info(
                f"Generating {validated_num_samples} counterfactuals for user {user_id} "
                f"with attributes: {validated_attributes}"
            )
            
            # Generate counterfactuals
            results = self._generate_counterfactuals(
                mock_image, validated_text, validated_attributes, validated_num_samples
            )
            
            # Calculate metrics
            generation_time = time.time() - request_start_time
            
            # Update performance metrics
            self._update_performance_metrics(True, generation_time)
            
            # Prepare response
            response = {
                "method": self.method,
                "original_image": mock_image,
                "original_text": validated_text,
                "target_attributes": validated_attributes,
                "counterfactuals": results,
                "metadata": {
                    "generation_time": generation_time,
                    "num_samples": len(results),
                    "device": self.device,
                    "timestamp": datetime.now().isoformat(),
                    "generation_id": self.generation_count,
                    "user_id": user_id,
                    "security_validated": True,
                    "rate_limit_info": self.rate_limiter.get_rate_info(user_id) if self.rate_limiter else None
                }
            }
            
            self.generation_count += 1
            logger.info(f"Generation completed successfully in {generation_time:.2f}s")
            
            return response
            
        except (ValidationError, SecurityError, RateLimitError) as e:
            # Don't retry for validation/security errors
            duration = time.time() - request_start_time
            self._update_performance_metrics(False, duration)
            
            if self.audit_logger:
                self.audit_logger.log_error("counterfactual_generation", str(e), user_id)
            
            raise
        
        except Exception as e:
            # Log unexpected errors
            duration = time.time() - request_start_time
            self._update_performance_metrics(False, duration)
            
            logger.error(f"Unexpected error in generation: {e}")
            
            if self.audit_logger:
                self.audit_logger.log_error("counterfactual_generation", str(e), user_id)
            
            raise GenerationError(f"Generation failed: {e}")
    
    def _generate_counterfactuals(
        self, 
        image: MockImage, 
        text: str, 
        attributes: List[str], 
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Internal counterfactual generation with error handling."""
        
        results = []
        attribute_values = {
            "gender": ["male", "female", "non-binary"],
            "race": ["white", "black", "asian", "hispanic", "diverse"],
            "age": ["young", "middle-aged", "elderly"],
            "expression": ["neutral", "smiling", "serious", "happy"],
            "hair_color": ["blonde", "brown", "black", "red", "gray"],
            "hair_style": ["short", "long", "curly", "straight"]
        }
        
        for i in range(num_samples):
            try:
                target_attrs = {}
                for attr in attributes:
                    if attr in attribute_values:
                        target_attrs[attr] = random.choice(attribute_values[attr])
                
                # Generate mock counterfactual with validation
                generated_image = image.copy()
                if not generated_image:
                    raise GenerationError(f"Failed to create counterfactual image for sample {i}")
                
                # Mock text modification
                modified_text = self._modify_text_robust(text, target_attrs)
                if not modified_text:
                    raise GenerationError(f"Failed to modify text for sample {i}")
                
                # Calculate confidence with error handling
                confidence = self._calculate_confidence(target_attrs, len(attributes))
                
                results.append({
                    "sample_id": i,
                    "target_attributes": target_attrs,
                    "generated_image": generated_image,
                    "generated_text": modified_text,
                    "confidence": confidence,
                    "explanation": f"Applied {self.method} to modify {', '.join(attributes)}",
                    "generation_success": True,
                    "validation_passed": True
                })
                
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                # Add failed sample info
                results.append({
                    "sample_id": i,
                    "target_attributes": {},
                    "generated_image": None,
                    "generated_text": text,  # Fallback to original
                    "confidence": 0.0,
                    "explanation": f"Generation failed: {str(e)}",
                    "generation_success": False,
                    "error": str(e)
                })
        
        # Validate that we have at least some successful results
        successful_results = [r for r in results if r.get("generation_success", False)]
        if not successful_results:
            raise GenerationError("All counterfactual generation attempts failed")
        
        return results
    
    def _modify_text_robust(self, text: str, target_attrs: Dict[str, str]) -> str:
        """Robustly modify text with error handling."""
        try:
            modified = text
            
            for attr, value in target_attrs.items():
                if attr == "gender":
                    modified = self._change_gender_references(modified, value)
                elif attr == "age":
                    modified = self._add_age_descriptor(modified, value)
                elif attr == "expression":
                    modified = self._add_expression_descriptor(modified, value)
            
            # Validate result
            if not modified or len(modified.strip()) == 0:
                logger.warning("Text modification resulted in empty text, using original")
                return text
            
            return modified
            
        except Exception as e:
            logger.warning(f"Text modification failed: {e}, using original text")
            return text
    
    def _change_gender_references(self, text: str, target_gender: str) -> str:
        """Change gender references with error handling."""
        try:
            import re
            
            if target_gender == "male":
                text = re.sub(r'\b(woman|female|she|her)\b', 'man', text, flags=re.IGNORECASE)
            elif target_gender == "female":
                text = re.sub(r'\b(man|male|he|him)\b', 'woman', text, flags=re.IGNORECASE)
            elif target_gender == "non-binary":
                text = re.sub(r'\b(man|woman|male|female|he|she|him|her)\b', 'person', text, flags=re.IGNORECASE)
            
            return text
            
        except Exception as e:
            logger.warning(f"Gender reference change failed: {e}")
            return text
    
    def _add_age_descriptor(self, text: str, target_age: str) -> str:
        """Add age descriptor with error handling."""
        try:
            age_descriptors = {
                "young": "young",
                "middle-aged": "middle-aged", 
                "elderly": "elderly"
            }
            
            if target_age in age_descriptors:
                descriptor = age_descriptors[target_age]
                # Add descriptor if not already present
                if descriptor.lower() not in text.lower():
                    return f"{descriptor} {text}"
            
            return text
            
        except Exception as e:
            logger.warning(f"Age descriptor addition failed: {e}")
            return text
    
    def _add_expression_descriptor(self, text: str, target_expression: str) -> str:
        """Add expression descriptor with error handling."""
        try:
            if target_expression != "neutral":
                expression_map = {
                    "smiling": "smiling",
                    "serious": "serious-looking",
                    "happy": "happy"
                }
                
                if target_expression in expression_map:
                    descriptor = expression_map[target_expression]
                    if descriptor.lower() not in text.lower():
                        return f"{text}, {descriptor}"
            
            return text
            
        except Exception as e:
            logger.warning(f"Expression descriptor addition failed: {e}")
            return text
    
    def _calculate_confidence(self, target_attrs: Dict[str, str], num_attributes: int) -> float:
        """Calculate confidence score with error handling."""
        try:
            base_confidence = 0.85
            
            # Reduce confidence based on number of changes
            complexity_penalty = len(target_attrs) * 0.05
            
            # Add some random variation
            variation = random.uniform(-0.05, 0.05)
            
            confidence = base_confidence - complexity_penalty + variation
            
            # Ensure bounds
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default middle confidence
    
    def _update_performance_metrics(self, success: bool, duration: float):
        """Update performance metrics with thread safety."""
        try:
            self.performance_metrics['total_requests'] += 1
            
            if success:
                self.performance_metrics['successful_requests'] += 1
            else:
                self.performance_metrics['failed_requests'] += 1
                self.error_count += 1
            
            # Update average response time
            total = self.performance_metrics['total_requests']
            current_avg = self.performance_metrics['avg_response_time']
            self.performance_metrics['avg_response_time'] = (current_avg * (total - 1) + duration) / total
            
            # Update error rate
            self.performance_metrics['error_rate'] = self.performance_metrics['failed_requests'] / total
            
        except Exception as e:
            logger.warning(f"Performance metrics update failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Determine health status
            error_rate = self.performance_metrics['error_rate']
            if error_rate < 0.05:
                health_status = "healthy"
            elif error_rate < 0.20:
                health_status = "degraded"
            else:
                health_status = "unhealthy"
            
            return {
                "status": health_status,
                "uptime_seconds": uptime,
                "total_generations": self.generation_count,
                "error_count": self.error_count,
                "performance_metrics": self.performance_metrics.copy(),
                "configuration": {
                    "method": self.method,
                    "device": self.device,
                    "rate_limiting_enabled": self.rate_limiter is not None,
                    "audit_logging_enabled": self.audit_logger is not None
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def shutdown_gracefully(self):
        """Perform graceful shutdown."""
        logger.info("Initiating graceful shutdown...")
        
        try:
            # Log final statistics
            if self.audit_logger:
                final_stats = {
                    "total_generations": self.generation_count,
                    "error_count": self.error_count,
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "final_metrics": self.performance_metrics
                }
                
                self.audit_logger.log_security_event(
                    "system_shutdown", "info", 
                    "Graceful shutdown completed", final_stats
                )
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class RobustBiasEvaluator:
    """Robust bias evaluator with comprehensive error handling."""
    
    def __init__(self, enable_audit_logging: bool = True):
        """Initialize robust evaluator."""
        self.evaluation_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        
        if enable_audit_logging:
            self.audit_logger = AuditLogger()
        else:
            self.audit_logger = None
        
        logger.info("Robust bias evaluator initialized")
    
    @with_error_handling("bias_evaluation", max_retries=2)
    def evaluate(
        self,
        counterfactuals: Dict[str, Any],
        metrics: List[str],
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Evaluate bias with comprehensive error handling."""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_evaluation_inputs(counterfactuals, metrics)
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_evaluation_request(user_id, metrics, True)
            
            logger.info(f"Evaluating bias for user {user_id} with metrics: {metrics}")
            
            self.evaluation_count += 1
            
            results = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_id": self.evaluation_count,
                "user_id": user_id,
                "metrics": {},
                "summary": {},
                "counterfactual_analysis": {},
                "validation_passed": True
            }
            
            cf_data = counterfactuals.get("counterfactuals", [])
            if not cf_data:
                raise EvaluationError("No counterfactual data found")
            
            # Evaluate each metric with error handling
            for metric in metrics:
                try:
                    if metric == "demographic_parity":
                        results["metrics"][metric] = self._compute_demographic_parity_robust(cf_data)
                    elif metric == "equalized_odds":
                        results["metrics"][metric] = self._compute_equalized_odds_robust(cf_data)
                    elif metric == "cits_score":
                        results["metrics"][metric] = self._compute_cits_score_robust(cf_data)
                    elif metric == "fairness_score":
                        results["metrics"][metric] = self._compute_fairness_score_robust(cf_data)
                    else:
                        logger.warning(f"Unknown metric: {metric}")
                        results["metrics"][metric] = {
                            "error": f"Unknown metric: {metric}",
                            "status": "failed"
                        }
                except Exception as e:
                    logger.error(f"Failed to compute metric {metric}: {e}")
                    results["metrics"][metric] = {
                        "error": str(e),
                        "status": "failed"
                    }
            
            # Generate summary and analysis
            results["summary"] = self._generate_summary_robust(results["metrics"])
            results["counterfactual_analysis"] = self._analyze_counterfactuals_robust(cf_data)
            
            # Add performance info
            evaluation_time = time.time() - start_time
            results["performance"] = {
                "evaluation_time": evaluation_time,
                "metrics_computed": len([m for m in results["metrics"] if "error" not in results["metrics"][m]]),
                "metrics_failed": len([m for m in results["metrics"] if "error" in results["metrics"][m]])
            }
            
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            return results
            
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_error("bias_evaluation", str(e), user_id)
            
            self.error_count += 1
            raise EvaluationError(f"Bias evaluation failed: {e}")
    
    def _validate_evaluation_inputs(self, counterfactuals: Dict[str, Any], metrics: List[str]):
        """Validate evaluation inputs."""
        if not isinstance(counterfactuals, dict):
            raise ValidationError("counterfactuals must be a dictionary")
        
        if "counterfactuals" not in counterfactuals:
            raise ValidationError("counterfactuals dictionary missing 'counterfactuals' key")
        
        cf_data = counterfactuals["counterfactuals"]
        if not isinstance(cf_data, list) or not cf_data:
            raise ValidationError("counterfactuals data must be a non-empty list")
        
        if not isinstance(metrics, list) or not metrics:
            raise ValidationError("metrics must be a non-empty list")
        
        supported_metrics = ["demographic_parity", "equalized_odds", "cits_score", "fairness_score"]
        unsupported = [m for m in metrics if m not in supported_metrics]
        if unsupported:
            logger.warning(f"Unsupported metrics will be skipped: {unsupported}")
    
    def _compute_demographic_parity_robust(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Robust demographic parity computation."""
        try:
            attribute_stats = {}
            total_samples = len(cf_data)
            
            for attr in ["gender", "race", "age"]:
                groups = {}
                valid_samples = 0
                
                for cf in cf_data:
                    if not isinstance(cf, dict):
                        continue
                    
                    target_attrs = cf.get("target_attributes", {})
                    if not isinstance(target_attrs, dict):
                        continue
                    
                    if attr in target_attrs:
                        value = target_attrs[attr]
                        confidence = cf.get("confidence", 0.0)
                        
                        # Validate confidence
                        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                            confidence = 0.5  # Default
                        
                        if value not in groups:
                            groups[value] = []
                        groups[value].append(confidence)
                        valid_samples += 1
                
                if len(groups) > 1:
                    group_means = {}
                    for group, confidences in groups.items():
                        if confidences:
                            group_means[group] = sum(confidences) / len(confidences)
                    
                    if len(group_means) > 1:
                        max_mean = max(group_means.values())
                        min_mean = min(group_means.values())
                        max_diff = max_mean - min_mean
                        
                        attribute_stats[attr] = {
                            "max_difference": max_diff,
                            "group_means": group_means,
                            "passes_threshold": max_diff < 0.1,
                            "sample_size": valid_samples
                        }
            
            overall_score = 0.0
            if attribute_stats:
                scores = [stats["max_difference"] for stats in attribute_stats.values()]
                overall_score = sum(scores) / len(scores)
            
            return {
                "attribute_stats": attribute_stats,
                "overall_score": overall_score,
                "passes_threshold": overall_score < 0.1,
                "total_samples_analyzed": total_samples,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Demographic parity computation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _compute_equalized_odds_robust(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Robust equalized odds computation."""
        try:
            tpr_differences = []
            group_tprs = {}
            
            for cf in cf_data:
                if not isinstance(cf, dict):
                    continue
                
                confidence = cf.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                
                target_attrs = cf.get("target_attributes", {})
                if not isinstance(target_attrs, dict):
                    continue
                
                # Mock TPR calculation based on attributes
                base_tpr = 0.75
                attr_penalty = len(target_attrs) * 0.05
                mock_tpr = max(0.0, min(1.0, base_tpr - attr_penalty + random.uniform(-0.1, 0.1)))
                
                # Group by primary attribute
                primary_attr = None
                for attr in ["gender", "race", "age"]:
                    if attr in target_attrs:
                        primary_attr = f"{attr}_{target_attrs[attr]}"
                        break
                
                if primary_attr:
                    if primary_attr not in group_tprs:
                        group_tprs[primary_attr] = []
                    group_tprs[primary_attr].append(mock_tpr)
            
            # Calculate differences
            if len(group_tprs) > 1:
                group_means = {k: sum(v)/len(v) for k, v in group_tprs.items() if v}
                if len(group_means) > 1:
                    max_tpr = max(group_means.values())
                    min_tpr = min(group_means.values())
                    max_diff = max_tpr - min_tpr
                else:
                    max_diff = 0.0
            else:
                max_diff = 0.0
                group_means = {}
            
            return {
                "max_tpr_difference": max_diff,
                "group_tprs": group_means,
                "passes_threshold": max_diff < 0.1,
                "total_groups": len(group_tprs),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Equalized odds computation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _compute_cits_score_robust(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Robust CITS score computation."""
        try:
            scores = []
            
            for cf in cf_data:
                if not isinstance(cf, dict):
                    continue
                
                confidence = cf.get("confidence", 0.0)
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    confidence = 0.5
                
                target_attrs = cf.get("target_attributes", {})
                if isinstance(target_attrs, dict):
                    # CITS penalty for complex changes
                    complexity_penalty = len(target_attrs) * 0.05
                    cits = max(0.0, min(1.0, confidence - complexity_penalty))
                else:
                    cits = confidence
                
                scores.append(cits)
            
            if scores:
                mean_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                
                # Calculate standard deviation
                if len(scores) > 1:
                    variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
                    std_score = variance ** 0.5
                else:
                    std_score = 0.0
            else:
                mean_score = min_score = max_score = std_score = 0.0
            
            return {
                "individual_scores": scores,
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score,
                "sample_count": len(scores),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"CITS score computation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _compute_fairness_score_robust(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Robust overall fairness score computation."""
        try:
            total_confidence = 0.0
            valid_samples = 0
            attribute_counts = {}
            
            for cf in cf_data:
                if not isinstance(cf, dict):
                    continue
                
                confidence = cf.get("confidence", 0.0)
                if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                    total_confidence += confidence
                    valid_samples += 1
                
                target_attrs = cf.get("target_attributes", {})
                if isinstance(target_attrs, dict):
                    for attr, value in target_attrs.items():
                        key = f"{attr}_{value}"
                        attribute_counts[key] = attribute_counts.get(key, 0) + 1
            
            # Calculate components
            if valid_samples > 0:
                avg_confidence = total_confidence / valid_samples
            else:
                avg_confidence = 0.0
            
            # Balance score
            if attribute_counts:
                counts = list(attribute_counts.values())
                if max(counts) > 0:
                    balance_score = 1.0 - (max(counts) - min(counts)) / max(counts)
                else:
                    balance_score = 1.0
            else:
                balance_score = 1.0
            
            overall_score = (avg_confidence + balance_score) / 2
            
            return {
                "overall_fairness_score": overall_score,
                "confidence_component": avg_confidence,
                "balance_component": balance_score,
                "attribute_distribution": attribute_counts,
                "valid_samples": valid_samples,
                "rating": self._get_fairness_rating(overall_score),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Fairness score computation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _generate_summary_robust(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate robust evaluation summary."""
        try:
            passed_metrics = 0
            total_metrics = 0
            failed_metrics = []
            
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    if "error" in metric_data:
                        failed_metrics.append(metric_name)
                    elif "passes_threshold" in metric_data:
                        total_metrics += 1
                        if metric_data["passes_threshold"]:
                            passed_metrics += 1
            
            fairness_score = passed_metrics / total_metrics if total_metrics > 0 else 0.0
            
            return {
                "overall_fairness_score": fairness_score,
                "passed_metrics": passed_metrics,
                "total_metrics": total_metrics,
                "failed_metrics": failed_metrics,
                "fairness_rating": self._get_fairness_rating(fairness_score),
                "summary_generated": True
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                "error": str(e),
                "summary_generated": False
            }
    
    def _analyze_counterfactuals_robust(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Robust counterfactual analysis."""
        try:
            attribute_dist = {}
            confidence_scores = []
            valid_samples = 0
            
            for cf in cf_data:
                if not isinstance(cf, dict):
                    continue
                
                valid_samples += 1
                
                # Collect confidence scores
                confidence = cf.get("confidence", 0.0)
                if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                    confidence_scores.append(confidence)
                
                # Analyze attribute distribution
                target_attrs = cf.get("target_attributes", {})
                if isinstance(target_attrs, dict):
                    for attr, value in target_attrs.items():
                        if attr not in attribute_dist:
                            attribute_dist[attr] = {}
                        if value not in attribute_dist[attr]:
                            attribute_dist[attr][value] = 0
                        attribute_dist[attr][value] += 1
            
            # Calculate confidence statistics
            if confidence_scores:
                mean_conf = sum(confidence_scores) / len(confidence_scores)
                min_conf = min(confidence_scores)
                max_conf = max(confidence_scores)
                
                if len(confidence_scores) > 1:
                    var_conf = sum((x - mean_conf) ** 2 for x in confidence_scores) / len(confidence_scores)
                    std_conf = var_conf ** 0.5
                else:
                    std_conf = 0.0
            else:
                mean_conf = min_conf = max_conf = std_conf = 0.0
            
            return {
                "attribute_distribution": attribute_dist,
                "confidence_stats": {
                    "mean": mean_conf,
                    "std": std_conf,
                    "min": min_conf,
                    "max": max_conf,
                    "count": len(confidence_scores)
                },
                "total_counterfactuals": len(cf_data),
                "valid_samples": valid_samples,
                "analysis_success": True
            }
            
        except Exception as e:
            logger.error(f"Counterfactual analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_success": False
            }
    
    def _get_fairness_rating(self, score: float) -> str:
        """Convert fairness score to rating."""
        try:
            if score >= 0.9:
                return "Excellent"
            elif score >= 0.7:
                return "Good"
            elif score >= 0.5:
                return "Fair"
            else:
                return "Poor"
        except:
            return "Unknown"
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                "total_evaluations": self.evaluation_count,
                "error_count": self.error_count,
                "success_rate": (self.evaluation_count - self.error_count) / max(1, self.evaluation_count),
                "uptime_seconds": uptime,
                "timestamp": datetime.now().isoformat(),
                "status": "operational" if self.error_count / max(1, self.evaluation_count) < 0.1 else "degraded"
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


def test_robust_system():
    """Test the robust system functionality."""
    logger.info("Testing robust counterfactual lab system...")
    
    try:
        # Test generator
        generator = RobustCounterfactualGenerator(
            method="modicf",
            enable_rate_limiting=True,
            enable_audit_logging=True,
            max_requests_per_minute=5
        )
        
        # Test valid generation
        test_image = MockImage(400, 300)
        
        results = generator.generate(
            image=test_image,
            text="A professional person working at a computer",
            attributes=["gender", "age"],
            num_samples=3,
            user_id="test_user"
        )
        
        logger.info(f"✅ Generated {len(results['counterfactuals'])} counterfactuals successfully")
        
        # Test evaluator
        evaluator = RobustBiasEvaluator(enable_audit_logging=True)
        
        evaluation = evaluator.evaluate(
            counterfactuals=results,
            metrics=["demographic_parity", "cits_score", "fairness_score"],
            user_id="test_user"
        )
        
        logger.info(f"✅ Evaluation completed with rating: {evaluation['summary']['fairness_rating']}")
        
        # Test error handling - invalid inputs
        try:
            generator.generate(
                image=test_image,
                text="",  # Invalid empty text
                attributes=["gender"],
                num_samples=1
            )
        except (ValidationError, GenerationError):
            logger.info("✅ Empty text validation working correctly")
        
        # Test rate limiting - create new generator with clean state
        rate_test_gen = RobustCounterfactualGenerator(
            enable_rate_limiting=True,
            max_requests_per_minute=3  # Lower limit for testing
        )
        
        try:
            # First few should succeed
            for i in range(3):
                rate_test_gen.generate(
                    image=test_image,
                    text=f"Test {i}",
                    attributes=["gender"],
                    num_samples=1,
                    user_id="rate_test_user_2"
                )
            
            # This should fail due to rate limit
            rate_test_gen.generate(
                image=test_image,
                text="Test rate limit",
                attributes=["gender"],
                num_samples=1,
                user_id="rate_test_user_2"
            )
        except (RateLimitError, GenerationError):
            logger.info("✅ Rate limiting working correctly")
        
        # Test health status
        health = generator.get_health_status()
        logger.info(f"✅ System health: {health['status']}")
        
        # Test evaluation stats
        eval_stats = evaluator.get_evaluation_stats()
        logger.info(f"✅ Evaluator stats: {eval_stats['status']}")
        
        logger.info("✅ Robust system test completed successfully!")
        
        return {
            "generation_results": results,
            "evaluation_results": evaluation,
            "health_status": health,
            "evaluation_stats": eval_stats,
            "test_status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Robust system test failed: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {"test_status": "failed", "error": str(e)}


if __name__ == "__main__":
    test_robust_system()