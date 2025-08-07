"""Custom exceptions for the Multimodal Counterfactual Lab."""


class CounterfactualLabError(Exception):
    """Base exception for all counterfactual lab errors."""
    pass


class GenerationError(CounterfactualLabError):
    """Raised when counterfactual generation fails."""
    pass


class ModelInitializationError(CounterfactualLabError):
    """Raised when model initialization fails."""
    pass


class ValidationError(CounterfactualLabError):
    """Raised when input validation fails."""
    pass


class CacheError(CounterfactualLabError):
    """Raised when cache operations fail."""
    pass


class StorageError(CounterfactualLabError):
    """Raised when storage operations fail."""
    pass


class EvaluationError(CounterfactualLabError):
    """Raised when bias evaluation fails."""
    pass


class AttributeError(CounterfactualLabError):
    """Raised when attribute operations fail."""
    pass


class DeviceError(CounterfactualLabError):
    """Raised when device operations fail."""
    pass


class ConfigurationError(CounterfactualLabError):
    """Raised when configuration is invalid."""
    pass