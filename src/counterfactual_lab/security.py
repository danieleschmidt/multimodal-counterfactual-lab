"""Security utilities and input sanitization for counterfactual generation."""

import re
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sqlite3
from PIL import Image
import secrets
import time

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Comprehensive security validation and sanitization."""
    
    # Dangerous file patterns
    DANGEROUS_PATTERNS = [
        r'\.exe$', r'\.bat$', r'\.cmd$', r'\.sh$', r'\.ps1$',
        r'\.php$', r'\.asp$', r'\.jsp$', r'\.js$', r'\.html$',
        r'\.py$', r'\.rb$', r'\.pl$'
    ]
    
    # Suspicious text patterns
    SUSPICIOUS_TEXT_PATTERNS = [
        r'<script', r'javascript:', r'onload=', r'onerror=',
        r'eval\(', r'exec\(', r'import\s+', r'from\s+.*\s+import',
        r'__.*__', r'\.system\(', r'os\.', r'subprocess\.',
        r'eval\s*\(', r'compile\s*\(', r'globals\s*\(',
        r'locals\s*\(', r'vars\s*\(', r'dir\s*\('
    ]
    
    # Rate limiting storage
    _rate_limits: Dict[str, List[float]] = {}
    
    @classmethod
    def sanitize_text_input(cls, text: str, max_length: int = 2000) -> str:
        """Sanitize text input for safe processing."""
        if not isinstance(text, str):
            raise ValueError("Input must be string")
        
        # Basic length check
        if len(text) > max_length:
            logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_TEXT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in text: {pattern}")
                # Don't reject, but log the warning
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Basic HTML entity escaping for safety
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('"', '&quot;').replace("'", '&#x27;')
        
        return text
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> Tuple[bool, str]:
        """Validate file path for security issues."""
        if not isinstance(file_path, str):
            return False, "Path must be string"
        
        # Convert to Path object
        try:
            path = Path(file_path)
        except Exception as e:
            return False, f"Invalid path: {e}"
        
        # Check for path traversal
        if '..' in str(path) or str(path).startswith('/'):
            return False, "Path traversal detected"
        
        # Check for dangerous file extensions
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, str(path), re.IGNORECASE):
                return False, f"Dangerous file pattern: {pattern}"
        
        # Check absolute path length
        if len(str(path.resolve())) > 4096:
            return False, "Path too long"
        
        return True, "Valid"
    
    @classmethod
    def validate_image_file(cls, image_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate image file for security and integrity."""
        is_valid, message = cls.validate_file_path(image_path)
        if not is_valid:
            return False, message, {}
        
        try:
            path = Path(image_path)
            
            if not path.exists():
                return False, "File not found", {}
            
            # Check file size (max 50MB)
            file_size = path.stat().st_size
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                return False, f"File too large: {file_size} bytes (max: {max_size})", {}
            
            # Validate image format and content
            try:
                with Image.open(path) as img:
                    img.verify()
                    
                # Reload for format check
                with Image.open(path) as img:
                    # Check dimensions
                    width, height = img.size
                    if width > 8192 or height > 8192:
                        return False, f"Image dimensions too large: {width}x{height}", {}
                    
                    if width < 16 or height < 16:
                        return False, f"Image dimensions too small: {width}x{height}", {}
                    
                    # Check format
                    allowed_formats = ['JPEG', 'PNG', 'BMP', 'TIFF']
                    if img.format not in allowed_formats:
                        return False, f"Unsupported format: {img.format}", {}
                    
                    metadata = {
                        'format': img.format,
                        'size': (width, height),
                        'mode': img.mode,
                        'file_size': file_size
                    }
                    
                    return True, "Valid image", metadata
                    
            except Exception as e:
                return False, f"Invalid image file: {e}", {}
                
        except Exception as e:
            return False, f"File validation error: {e}", {}
    
    @classmethod
    def check_rate_limit(cls, identifier: str, max_requests: int = 60, window_seconds: int = 60) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limiting for requests."""
        current_time = time.time()
        
        # Clean old entries
        if identifier in cls._rate_limits:
            cls._rate_limits[identifier] = [
                t for t in cls._rate_limits[identifier]
                if current_time - t < window_seconds
            ]
        else:
            cls._rate_limits[identifier] = []
        
        # Check current rate
        current_requests = len(cls._rate_limits[identifier])
        
        if current_requests >= max_requests:
            return False, {
                'allowed': False,
                'current_requests': current_requests,
                'max_requests': max_requests,
                'window_seconds': window_seconds,
                'retry_after': window_seconds - (current_time - min(cls._rate_limits[identifier]))
            }
        
        # Add current request
        cls._rate_limits[identifier].append(current_time)
        
        return True, {
            'allowed': True,
            'current_requests': current_requests + 1,
            'max_requests': max_requests,
            'window_seconds': window_seconds,
            'remaining_requests': max_requests - current_requests - 1
        }
    
    @classmethod
    def validate_sql_input(cls, input_str: str) -> Tuple[bool, str]:
        """Validate input for SQL injection patterns."""
        if not isinstance(input_str, str):
            return False, "Input must be string"
        
        # SQL injection patterns
        sql_patterns = [
            r';.*--', r'union.*select', r'drop.*table', r'delete.*from',
            r'insert.*into', r'update.*set', r'alter.*table', r'create.*table',
            r'exec.*\(', r'sp_.*', r'xp_.*', r'.*;\s*shutdown',
            r'benchmark\s*\(', r'sleep\s*\(', r'waitfor.*delay'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return False, f"Suspicious SQL pattern detected: {pattern}"
        
        return True, "Valid"
    
    @classmethod
    def generate_secure_id(cls, length: int = 32) -> str:
        """Generate cryptographically secure random ID."""
        return secrets.token_urlsafe(length)[:length]
    
    @classmethod
    def compute_file_hash(cls, file_path: str, algorithm: str = 'sha256') -> Optional[str]:
        """Compute secure hash of file."""
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            return None
    
    @classmethod
    def validate_json_input(cls, json_str: str, max_depth: int = 10, max_keys: int = 1000) -> Tuple[bool, str]:
        """Validate JSON input for security issues."""
        try:
            import json
            
            # Basic length check
            if len(json_str) > 1024 * 1024:  # 1MB limit
                return False, "JSON too large"
            
            # Parse and validate structure
            data = json.loads(json_str)
            
            # Check depth and key count
            def check_structure(obj, depth=0):
                if depth > max_depth:
                    raise ValueError("JSON too deeply nested")
                
                if isinstance(obj, dict):
                    if len(obj) > max_keys:
                        raise ValueError("Too many keys in JSON object")
                    for value in obj.values():
                        check_structure(value, depth + 1)
                elif isinstance(obj, list):
                    if len(obj) > max_keys:
                        raise ValueError("JSON array too large")
                    for item in obj:
                        check_structure(item, depth + 1)
            
            check_structure(data)
            return True, "Valid JSON"
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"JSON validation error: {e}"


class AuditLogger:
    """Security audit logging functionality."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup separate security logger
        self.logger = logging.getLogger('security_audit')
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_access_attempt(self, user_id: str, resource: str, success: bool, details: Dict[str, Any] = None):
        """Log access attempts."""
        details = details or {}
        self.logger.info(f"ACCESS: user={user_id} resource={resource} success={success} details={details}")
    
    def log_security_event(self, event_type: str, severity: str, description: str, details: Dict[str, Any] = None):
        """Log security events."""
        details = details or {}
        log_func = getattr(self.logger, severity.lower(), self.logger.info)
        log_func(f"SECURITY: type={event_type} desc={description} details={details}")
    
    def log_generation_request(self, user_id: str, method: str, attributes: List[str], success: bool):
        """Log counterfactual generation requests."""
        self.logger.info(f"GENERATION: user={user_id} method={method} attributes={attributes} success={success}")
    
    def log_evaluation_request(self, user_id: str, metrics: List[str], success: bool):
        """Log bias evaluation requests."""
        self.logger.info(f"EVALUATION: user={user_id} metrics={metrics} success={success}")


class SecureConfigManager:
    """Secure configuration management."""
    
    def __init__(self, config_file: str = "secure_config.json"):
        self.config_file = Path(config_file)
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration with validation."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    content = f.read()
                
                # Validate JSON
                is_valid, message = SecurityValidator.validate_json_input(content)
                if not is_valid:
                    logger.error(f"Invalid config JSON: {message}")
                    return
                
                import json
                self._config = json.loads(content)
                logger.info("Configuration loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
        self._save_config()
    
    def _save_config(self):
        """Save configuration securely."""
        try:
            import json
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            # Set restrictive permissions
            self.config_file.chmod(0o600)
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")


class SecureSessionManager:
    """Secure session management for multi-user scenarios."""
    
    def __init__(self, session_timeout: int = 3600):
        self.session_timeout = session_timeout
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, user_id: str) -> str:
        """Create secure session."""
        session_id = SecurityValidator.generate_secure_id(64)
        
        self._sessions[session_id] = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'requests_count': 0
        }
        
        logger.info(f"Session created for user {user_id}: {session_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """Validate session and return user_id if valid."""
        if session_id not in self._sessions:
            return False, None
        
        session = self._sessions[session_id]
        current_time = time.time()
        
        # Check timeout
        if current_time - session['last_activity'] > self.session_timeout:
            del self._sessions[session_id]
            logger.info(f"Session expired: {session_id}")
            return False, None
        
        # Update activity
        session['last_activity'] = current_time
        session['requests_count'] += 1
        
        return True, session['user_id']
    
    def invalidate_session(self, session_id: str):
        """Invalidate session."""
        if session_id in self._sessions:
            user_id = self._sessions[session_id]['user_id']
            del self._sessions[session_id]
            logger.info(f"Session invalidated for user {user_id}: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self._sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class SecurityMiddleware:
    """Security middleware for request processing."""
    
    def __init__(self, audit_logger: AuditLogger, session_manager: SecureSessionManager):
        self.audit_logger = audit_logger
        self.session_manager = session_manager
    
    def process_request(self, request_data: Dict[str, Any], session_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Process request with security checks."""
        result = {
            'allowed': False,
            'user_id': None,
            'errors': [],
            'warnings': []
        }
        
        # Session validation
        if session_id:
            is_valid, user_id = self.session_manager.validate_session(session_id)
            if not is_valid:
                result['errors'].append("Invalid or expired session")
                self.audit_logger.log_security_event(
                    'session_validation', 'warning', 
                    'Invalid session attempt', {'session_id': session_id}
                )
                return False, result
            result['user_id'] = user_id
        else:
            result['user_id'] = 'anonymous'
        
        # Rate limiting
        identifier = result['user_id']
        rate_ok, rate_info = SecurityValidator.check_rate_limit(identifier)
        if not rate_ok:
            result['errors'].append(f"Rate limit exceeded. Retry after {rate_info['retry_after']:.1f} seconds")
            self.audit_logger.log_security_event(
                'rate_limit', 'warning', 
                'Rate limit exceeded', {'user_id': identifier, 'rate_info': rate_info}
            )
            return False, result
        
        # Input validation
        if 'text' in request_data:
            try:
                request_data['text'] = SecurityValidator.sanitize_text_input(request_data['text'])
            except Exception as e:
                result['errors'].append(f"Text validation failed: {e}")
                return False, result
        
        if 'image_path' in request_data:
            is_valid, message, metadata = SecurityValidator.validate_image_file(request_data['image_path'])
            if not is_valid:
                result['errors'].append(f"Image validation failed: {message}")
                return False, result
            result['image_metadata'] = metadata
        
        # SQL input validation for database operations
        for key, value in request_data.items():
            if isinstance(value, str):
                is_valid, message = SecurityValidator.validate_sql_input(value)
                if not is_valid:
                    result['errors'].append(f"SQL validation failed for {key}: {message}")
                    return False, result
        
        result['allowed'] = True
        return True, result