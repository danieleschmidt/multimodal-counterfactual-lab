"""Enhanced core functionality with robust error handling and security."""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from PIL import Image

from counterfactual_lab.core import CounterfactualGenerator as BaseGenerator, BiasEvaluator as BaseEvaluator
from counterfactual_lab.error_handling import with_error_handling, ErrorContext, get_error_tracker, HealthChecker
from counterfactual_lab.security import SecurityValidator, AuditLogger, SecurityMiddleware, SecureSessionManager
from counterfactual_lab.exceptions import GenerationError, ValidationError, SecurityError

logger = logging.getLogger(__name__)


class EnhancedCounterfactualGenerator(BaseGenerator):
    """Enhanced counterfactual generator with security and error handling."""
    
    def __init__(self, method: str = "modicf", device: str = "cuda", 
                 enable_security: bool = True, session_id: Optional[str] = None,
                 **kwargs):
        """Initialize enhanced generator with security features."""
        
        # Initialize security components
        self.enable_security = enable_security
        self.session_id = session_id
        
        if enable_security:
            self.audit_logger = AuditLogger("generation_audit.log")
            self.session_manager = SecureSessionManager()
            self.security_middleware = SecurityMiddleware(self.audit_logger, self.session_manager)
        
        # Initialize health checker
        self.error_tracker = get_error_tracker()
        self.health_checker = HealthChecker(self.error_tracker)
        
        # Register health checks
        self._register_health_checks()
        
        try:
            # Initialize base generator
            super().__init__(method=method, device=device, **kwargs)
            logger.info(f"Enhanced generator initialized with security={'enabled' if enable_security else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced generator: {e}")
            raise
    
    def _register_health_checks(self):
        """Register health check functions."""
        
        def check_device_availability():
            """Check if compute device is available."""
            try:
                if self.device == "cuda":
                    try:
                        import torch
                        return torch.cuda.is_available()
                    except ImportError:
                        return False
                return True
            except Exception:
                return False
        
        def check_cache_health():
            """Check cache system health."""
            try:
                if self.cache_manager:
                    stats = self.cache_manager.get_cache_stats()
                    return stats.get('utilization_percent', 0) < 95  # Cache not too full
                return True
            except Exception:
                return False
        
        def check_storage_health():
            """Check storage system health."""
            try:
                if self.storage_manager:
                    stats = self.storage_manager.get_storage_stats()
                    return 'error' not in stats
                return True
            except Exception:
                return False
        
        self.health_checker.register_health_check('device', check_device_availability)
        self.health_checker.register_health_check('cache', check_cache_health)
        self.health_checker.register_health_check('storage', check_storage_health)
    
    @with_error_handling("secure_generate", max_retries=2, circuit_breaker_name="generation")
    def secure_generate(
        self,
        image: Union[str, Path, Image.Image],
        text: str,
        attributes: Union[List[str], str],
        num_samples: int = 5,
        user_id: Optional[str] = None,
        save_results: bool = False,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate counterfactuals with enhanced security and error handling."""
        
        context = ErrorContext("secure_generate", user_id=user_id)
        context.add_context(
            method=self.method,
            attributes=attributes,
            num_samples=num_samples
        )
        
        try:
            # Security validation
            if self.enable_security:
                request_data = {
                    'text': text,
                    'attributes': attributes,
                    'num_samples': num_samples
                }
                
                # Add image path if it's a file
                if isinstance(image, (str, Path)):
                    request_data['image_path'] = str(image)
                
                is_allowed, security_result = self.security_middleware.process_request(
                    request_data, self.session_id
                )
                
                if not is_allowed:
                    error_msg = "; ".join(security_result['errors'])
                    self.audit_logger.log_security_event(
                        'generation_blocked', 'warning',
                        f'Generation request blocked: {error_msg}',
                        {'user_id': user_id, 'errors': security_result['errors']}
                    )
                    raise SecurityError(f"Security validation failed: {error_msg}")
                
                # Use sanitized text from security validation
                text = request_data.get('text', text)
                user_id = security_result.get('user_id', user_id)
            
            # Input validation with enhanced checks
            validated_image = self._enhanced_validate_image(image, context)
            validated_text = self._enhanced_validate_text(text, context)
            validated_attributes = self._enhanced_validate_attributes(attributes, context)
            validated_num_samples = self._enhanced_validate_num_samples(num_samples, context)
            
            # Log generation attempt
            if self.enable_security:
                self.audit_logger.log_generation_request(
                    user_id or 'anonymous', 
                    self.method, 
                    validated_attributes, 
                    True
                )
            
            # Perform generation using base class
            result = super().generate(
                image=validated_image,
                text=validated_text,
                attributes=validated_attributes,
                num_samples=validated_num_samples,
                save_results=save_results,
                experiment_id=experiment_id
            )
            
            # Add security metadata
            result['metadata']['security_enabled'] = self.enable_security
            result['metadata']['user_id'] = user_id
            result['metadata']['validation_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Secure generation completed successfully for user {user_id}")
            return result
            
        except SecurityError:
            raise
        except Exception as e:
            if self.enable_security:
                self.audit_logger.log_generation_request(
                    user_id or 'anonymous', 
                    self.method, 
                    getattr(attributes, '__iter__', lambda: [])() if hasattr(attributes, '__iter__') else [str(attributes)], 
                    False
                )
            
            logger.error(f"Secure generation failed for user {user_id}: {e}")
            raise GenerationError(f"Secure generation failed: {e}")
    
    def _enhanced_validate_image(self, image: Union[str, Path, Image.Image], context: ErrorContext) -> Image.Image:
        """Enhanced image validation with security checks."""
        try:
            # File-based validation
            if isinstance(image, (str, Path)):
                # Security validation
                is_valid, message, metadata = SecurityValidator.validate_image_file(str(image))
                if not is_valid:
                    raise ValidationError(f"Image security validation failed: {message}")
                
                context.add_context(image_metadata=metadata)
                
                # Load and validate
                validated_image = Image.open(image).convert("RGB")
                validated_image.verify()
                
                # Reload after verify
                validated_image = Image.open(image).convert("RGB")
                
            else:
                # Direct PIL Image
                validated_image = image.convert("RGB")
            
            # Additional size and content checks
            width, height = validated_image.size
            
            if width * height > 16777216:  # 4096x4096
                logger.warning(f"Large image detected: {width}x{height}")
            
            context.add_context(image_size=(width, height))
            
            return validated_image
            
        except Exception as e:
            raise ValidationError(f"Enhanced image validation failed: {e}")
    
    def _enhanced_validate_text(self, text: str, context: ErrorContext) -> str:
        """Enhanced text validation with security checks."""
        try:
            # Security sanitization
            sanitized_text = SecurityValidator.sanitize_text_input(text, max_length=2000)
            
            # SQL injection check
            is_safe, message = SecurityValidator.validate_sql_input(sanitized_text)
            if not is_safe:
                raise ValidationError(f"Text contains suspicious patterns: {message}")
            
            context.add_context(text_length=len(sanitized_text))
            
            return sanitized_text
            
        except Exception as e:
            raise ValidationError(f"Enhanced text validation failed: {e}")
    
    def _enhanced_validate_attributes(self, attributes: Union[List[str], str], context: ErrorContext) -> List[str]:
        """Enhanced attribute validation."""
        try:
            # Convert to list if needed
            if isinstance(attributes, str):
                attr_list = [attr.strip().lower() for attr in attributes.split(",")]
            elif isinstance(attributes, list):
                attr_list = [str(attr).strip().lower() for attr in attributes]
            else:
                raise ValidationError("Attributes must be string or list")
            
            # Remove empty attributes
            attr_list = [attr for attr in attr_list if attr]
            
            if not attr_list:
                raise ValidationError("At least one attribute must be specified")
            
            # Validate against supported attributes
            SUPPORTED_ATTRIBUTES = {
                "gender": ["male", "female", "non-binary"],
                "race": ["white", "black", "asian", "hispanic", "diverse"],
                "age": ["young", "middle-aged", "elderly", "child", "adult"],
                "expression": ["neutral", "smiling", "serious", "happy", "sad"]
            }
            
            unsupported = [attr for attr in attr_list if attr not in SUPPORTED_ATTRIBUTES]
            if unsupported:
                raise ValidationError(f"Unsupported attributes: {unsupported}")
            
            context.add_context(validated_attributes=attr_list)
            
            return attr_list
            
        except Exception as e:
            raise ValidationError(f"Enhanced attribute validation failed: {e}")
    
    def _enhanced_validate_num_samples(self, num_samples: int, context: ErrorContext) -> int:
        """Enhanced num_samples validation."""
        try:
            if not isinstance(num_samples, int):
                num_samples = int(num_samples)
            
            if num_samples < 1:
                raise ValidationError("num_samples must be positive")
            
            if num_samples > 20:  # Lower limit for security
                raise ValidationError(f"num_samples too large: {num_samples} (max: 20)")
            
            context.add_context(num_samples=num_samples)
            
            return num_samples
            
        except Exception as e:
            raise ValidationError(f"Enhanced num_samples validation failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return self.health_checker.get_system_health_summary()
    
    def create_session(self, user_id: str) -> str:
        """Create secure session for user."""
        if not self.enable_security:
            logger.warning("Security not enabled, session creation ignored")
            return ""
        
        session_id = self.session_manager.create_session(user_id)
        self.audit_logger.log_security_event(
            'session_created', 'info',
            f'Session created for user {user_id}',
            {'session_id': session_id}
        )
        return session_id
    
    def invalidate_session(self, session_id: str):
        """Invalidate user session."""
        if not self.enable_security:
            return
        
        self.session_manager.invalidate_session(session_id)


class EnhancedBiasEvaluator(BaseEvaluator):
    """Enhanced bias evaluator with security and error handling."""
    
    def __init__(self, model, enable_security: bool = True, session_id: Optional[str] = None):
        """Initialize enhanced evaluator."""
        super().__init__(model)
        
        self.enable_security = enable_security
        self.session_id = session_id
        
        if enable_security:
            self.audit_logger = AuditLogger("evaluation_audit.log")
            self.session_manager = SecureSessionManager()
            self.security_middleware = SecurityMiddleware(self.audit_logger, self.session_manager)
        
        self.error_tracker = get_error_tracker()
        
        logger.info(f"Enhanced evaluator initialized with security={'enabled' if enable_security else 'disabled'}")
    
    @with_error_handling("secure_evaluate", max_retries=2, circuit_breaker_name="evaluation")
    def secure_evaluate(
        self,
        counterfactuals: Dict[str, Any],
        metrics: List[str],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate bias with enhanced security and error handling."""
        
        context = ErrorContext("secure_evaluate", user_id=user_id)
        context.add_context(
            metrics=metrics,
            num_counterfactuals=len(counterfactuals.get('counterfactuals', []))
        )
        
        try:
            # Security validation
            if self.enable_security:
                request_data = {
                    'metrics': ','.join(metrics),
                    'counterfactuals_count': len(counterfactuals.get('counterfactuals', []))
                }
                
                is_allowed, security_result = self.security_middleware.process_request(
                    request_data, self.session_id
                )
                
                if not is_allowed:
                    error_msg = "; ".join(security_result['errors'])
                    self.audit_logger.log_security_event(
                        'evaluation_blocked', 'warning',
                        f'Evaluation request blocked: {error_msg}',
                        {'user_id': user_id, 'errors': security_result['errors']}
                    )
                    raise SecurityError(f"Security validation failed: {error_msg}")
                
                user_id = security_result.get('user_id', user_id)
            
            # Validate inputs
            validated_metrics = self._validate_metrics(metrics, context)
            validated_counterfactuals = self._validate_counterfactuals(counterfactuals, context)
            
            # Log evaluation attempt
            if self.enable_security:
                self.audit_logger.log_evaluation_request(
                    user_id or 'anonymous', 
                    validated_metrics, 
                    True
                )
            
            # Perform evaluation using base class
            result = super().evaluate(validated_counterfactuals, validated_metrics)
            
            # Add security metadata
            result['security_metadata'] = {
                'security_enabled': self.enable_security,
                'user_id': user_id,
                'validation_timestamp': datetime.now().isoformat(),
                'metrics_validated': validated_metrics
            }
            
            logger.info(f"Secure evaluation completed successfully for user {user_id}")
            return result
            
        except SecurityError:
            raise
        except Exception as e:
            if self.enable_security:
                self.audit_logger.log_evaluation_request(
                    user_id or 'anonymous', 
                    metrics, 
                    False
                )
            
            logger.error(f"Secure evaluation failed for user {user_id}: {e}")
            raise GenerationError(f"Secure evaluation failed: {e}")
    
    def _validate_metrics(self, metrics: List[str], context: ErrorContext) -> List[str]:
        """Validate bias evaluation metrics."""
        SUPPORTED_METRICS = [
            "demographic_parity", "equalized_odds", "cits_score", 
            "disparate_impact", "statistical_parity_difference",
            "equal_opportunity_difference", "average_odds_difference"
        ]
        
        if not isinstance(metrics, list):
            raise ValidationError("Metrics must be a list")
        
        if not metrics:
            raise ValidationError("At least one metric must be specified")
        
        # Validate each metric
        unsupported = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
        if unsupported:
            raise ValidationError(f"Unsupported metrics: {unsupported}")
        
        context.add_context(validated_metrics=metrics)
        return metrics
    
    def _validate_counterfactuals(self, counterfactuals: Dict[str, Any], context: ErrorContext) -> Dict[str, Any]:
        """Validate counterfactuals data structure."""
        if not isinstance(counterfactuals, dict):
            raise ValidationError("Counterfactuals must be a dictionary")
        
        if 'counterfactuals' not in counterfactuals:
            raise ValidationError("Missing 'counterfactuals' key in data")
        
        cf_list = counterfactuals['counterfactuals']
        if not isinstance(cf_list, list):
            raise ValidationError("Counterfactuals must be a list")
        
        if not cf_list:
            raise ValidationError("No counterfactuals provided for evaluation")
        
        if len(cf_list) > 100:  # Limit for security
            raise ValidationError(f"Too many counterfactuals: {len(cf_list)} (max: 100)")
        
        context.add_context(counterfactual_count=len(cf_list))
        return counterfactuals