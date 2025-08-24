"""Input validation utilities for counterfactual generation."""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

from counterfactual_lab.exceptions import ValidationError, AttributeError

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates inputs for counterfactual generation and evaluation."""
    
    SUPPORTED_METHODS = ["modicf", "icg", "nacs-cf"]
    
    SUPPORTED_ATTRIBUTES = {
        "gender": ["male", "female", "non-binary"],
        "race": ["white", "black", "asian", "hispanic", "diverse"],
        "age": ["young", "middle-aged", "elderly", "child", "adult"],
        "expression": ["neutral", "smiling", "serious", "happy", "sad"],
        "hair_color": ["blonde", "brown", "black", "red", "gray", "white"],
        "hair_style": ["short", "long", "curly", "straight", "wavy"],
        "clothing": ["casual", "formal", "business", "medical", "uniform"],
        "background": ["indoor", "outdoor", "office", "medical", "neutral"]
    }
    
    SUPPORTED_DEVICES = ["cuda", "cpu", "auto"]
    SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    @classmethod
    def validate_method(cls, method: str) -> str:
        """Validate generation method."""
        if not isinstance(method, str):
            raise ValidationError(f"Method must be string, got {type(method)}")
        
        method = method.lower().strip()
        
        if method not in cls.SUPPORTED_METHODS:
            raise ValidationError(
                f"Unsupported method '{method}'. Supported: {cls.SUPPORTED_METHODS}"
            )
        
        return method
    
    @classmethod
    def validate_device(cls, device: str) -> str:
        """Validate device specification."""
        if not isinstance(device, str):
            raise ValidationError(f"Device must be string, got {type(device)}")
        
        device = device.lower().strip()
        
        if device not in cls.SUPPORTED_DEVICES:
            raise ValidationError(
                f"Unsupported device '{device}'. Supported: {cls.SUPPORTED_DEVICES}"
            )
        
        return device
    
    @classmethod
    def validate_image(cls, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Validate and load image."""
        try:
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                
                # Check if file exists
                if not image_path.exists():
                    raise ValidationError(f"Image file not found: {image_path}")
                
                # Check file extension
                if image_path.suffix.lower() not in cls.SUPPORTED_IMAGE_FORMATS:
                    raise ValidationError(
                        f"Unsupported image format '{image_path.suffix}'. "
                        f"Supported: {cls.SUPPORTED_IMAGE_FORMATS}"
                    )
                
                # Load image
                try:
                    pil_image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    raise ValidationError(f"Failed to load image {image_path}: {e}")
                
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            
            else:
                raise ValidationError(
                    f"Image must be string path, Path object, or PIL Image. Got {type(image)}"
                )
            
            # Validate image dimensions
            width, height = pil_image.size
            
            if width < 64 or height < 64:
                raise ValidationError(
                    f"Image too small: {width}x{height}. Minimum: 64x64"
                )
            
            if width > 4096 or height > 4096:
                logger.warning(f"Large image detected: {width}x{height}. May cause memory issues.")
            
            # Check if image is valid
            try:
                pil_image.verify()
                # Reload since verify() closes the file
                if isinstance(image, (str, Path)):
                    pil_image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValidationError(f"Image verification failed: {e}")
            
            return pil_image
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Image validation failed: {e}")
    
    @classmethod
    def validate_text(cls, text: str, min_length: int = 1, max_length: int = 1000) -> str:
        """Validate text input."""
        if not isinstance(text, str):
            raise ValidationError(f"Text must be string, got {type(text)}")
        
        text = text.strip()
        
        if len(text) < min_length:
            raise ValidationError(f"Text too short. Minimum length: {min_length}")
        
        if len(text) > max_length:
            raise ValidationError(f"Text too long. Maximum length: {max_length}")
        
        # Check for potentially problematic content
        problematic_patterns = [
            "\\x", "\\u", "\\n\\n\\n",  # Escape sequences or excessive newlines
        ]
        
        for pattern in problematic_patterns:
            if pattern in text:
                logger.warning(f"Text contains potentially problematic pattern: {pattern}")
        
        return text
    
    @classmethod
    def validate_attributes(cls, attributes: Union[str, List[str], Dict[str, str]]) -> List[str]:
        """Validate attribute list or dictionary."""
        if isinstance(attributes, str):
            # Split comma-separated string
            attr_list = [attr.strip().lower() for attr in attributes.split(",")]
        elif isinstance(attributes, list):
            attr_list = [str(attr).strip().lower() for attr in attributes]
        elif isinstance(attributes, dict):
            # For dictionary, just validate keys
            attr_list = [str(key).strip().lower() for key in attributes.keys()]
        else:
            raise ValidationError(
                f"Attributes must be string, list, or dict. Got {type(attributes)}"
            )
        
        # Remove empty attributes
        attr_list = [attr for attr in attr_list if attr]
        
        if not attr_list:
            raise ValidationError("At least one attribute must be specified")
        
        # Validate each attribute
        unsupported = []
        for attr in attr_list:
            if attr not in cls.SUPPORTED_ATTRIBUTES:
                unsupported.append(attr)
        
        if unsupported:
            raise AttributeError(
                f"Unsupported attributes: {unsupported}. "
                f"Supported: {list(cls.SUPPORTED_ATTRIBUTES.keys())}"
            )
        
        return attr_list
    
    @classmethod
    def validate_attribute_values(cls, attributes: Dict[str, str]) -> Dict[str, str]:
        """Validate attribute values."""
        if not isinstance(attributes, dict):
            raise ValidationError(f"Attributes must be dictionary, got {type(attributes)}")
        
        validated = {}
        
        for attr, value in attributes.items():
            attr = str(attr).strip().lower()
            value = str(value).strip().lower()
            
            if attr not in cls.SUPPORTED_ATTRIBUTES:
                raise AttributeError(f"Unsupported attribute: {attr}")
            
            if value not in cls.SUPPORTED_ATTRIBUTES[attr]:
                raise AttributeError(
                    f"Unsupported value '{value}' for attribute '{attr}'. "
                    f"Supported: {cls.SUPPORTED_ATTRIBUTES[attr]}"
                )
            
            validated[attr] = value
        
        return validated
    
    @classmethod
    def validate_num_samples(cls, num_samples: int, max_samples: int = 50) -> int:
        """Validate number of samples."""
        if not isinstance(num_samples, int):
            try:
                num_samples = int(num_samples)
            except (ValueError, TypeError):
                raise ValidationError(f"num_samples must be integer, got {type(num_samples)}")
        
        if num_samples < 1:
            raise ValidationError(f"num_samples must be positive, got {num_samples}")
        
        if num_samples > max_samples:
            raise ValidationError(
                f"num_samples too large: {num_samples}. Maximum: {max_samples}"
            )
        
        return num_samples
    
    @classmethod
    def validate_metrics(cls, metrics: Union[str, List[str]]) -> List[str]:
        """Validate bias evaluation metrics."""
        SUPPORTED_METRICS = [
            "demographic_parity", "equalized_odds", "cits_score", 
            "disparate_impact", "statistical_parity_difference",
            "equal_opportunity_difference", "average_odds_difference"
        ]
        
        if isinstance(metrics, str):
            metric_list = [metric.strip().lower() for metric in metrics.split(",")]
        elif isinstance(metrics, list):
            metric_list = [str(metric).strip().lower() for metric in metrics]
        else:
            raise ValidationError(f"Metrics must be string or list, got {type(metrics)}")
        
        # Remove empty metrics
        metric_list = [metric for metric in metric_list if metric]
        
        if not metric_list:
            raise ValidationError("At least one metric must be specified")
        
        # Validate each metric
        unsupported = []
        for metric in metric_list:
            if metric not in SUPPORTED_METRICS:
                unsupported.append(metric)
        
        if unsupported:
            raise ValidationError(
                f"Unsupported metrics: {unsupported}. Supported: {SUPPORTED_METRICS}"
            )
        
        return metric_list
    
    @classmethod
    def validate_file_path(cls, path: Union[str, Path], must_exist: bool = True,
                          extensions: Optional[List[str]] = None) -> Path:
        """Validate file path."""
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise ValidationError(f"Path must be string or Path object, got {type(path)}")
        
        if must_exist and not path.exists():
            raise ValidationError(f"File not found: {path}")
        
        if extensions:
            if path.suffix.lower() not in extensions:
                raise ValidationError(
                    f"Unsupported file extension '{path.suffix}'. "
                    f"Supported: {extensions}"
                )
        
        return path
    
    @classmethod
    def validate_directory(cls, path: Union[str, Path], create_if_missing: bool = False) -> Path:
        """Validate directory path."""
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise ValidationError(f"Path must be string or Path object, got {type(path)}")
        
        if not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {path}")
                except Exception as e:
                    raise ValidationError(f"Failed to create directory {path}: {e}")
            else:
                raise ValidationError(f"Directory not found: {path}")
        
        elif not path.is_dir():
            raise ValidationError(f"Path is not a directory: {path}")
        
        return path
    
    @classmethod
    def validate_experiment_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment configuration."""
        if not isinstance(config, dict):
            raise ValidationError(f"Config must be dictionary, got {type(config)}")
        
        validated_config = {}
        
        # Required fields
        required_fields = ["method", "attributes", "num_samples"]
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Required field missing: {field}")
        
        # Validate each field
        validated_config["method"] = cls.validate_method(config["method"])
        validated_config["attributes"] = cls.validate_attributes(config["attributes"])
        validated_config["num_samples"] = cls.validate_num_samples(config["num_samples"])
        
        # Optional fields
        if "device" in config:
            validated_config["device"] = cls.validate_device(config["device"])
        
        if "text" in config:
            validated_config["text"] = cls.validate_text(config["text"])
        
        if "save_results" in config:
            validated_config["save_results"] = bool(config["save_results"])
        
        if "use_cache" in config:
            validated_config["use_cache"] = bool(config["use_cache"])
        
        # Copy other fields as-is
        for key, value in config.items():
            if key not in validated_config:
                validated_config[key] = value
        
        return validated_config


class SafetyValidator:
    """Additional safety validations for responsible AI use."""
    
    @staticmethod
    def validate_ethical_use(text: str, attributes: List[str]) -> Tuple[bool, List[str]]:
        """Validate that the use case is ethical."""
        warnings = []
        
        # Check for potentially harmful text
        harmful_keywords = [
            "illegal", "violence", "hate", "discrimination", 
            "harassment", "exploit", "manipulate"
        ]
        
        text_lower = text.lower()
        for keyword in harmful_keywords:
            if keyword in text_lower:
                warnings.append(f"Text contains potentially harmful keyword: {keyword}")
        
        # Check for excessive demographic changes
        demographic_attrs = ["gender", "race", "age"]
        demographic_count = sum(1 for attr in attributes if attr in demographic_attrs)
        
        if demographic_count > 2:
            warnings.append(
                "Multiple demographic attribute changes detected. "
                "Ensure use case is for legitimate bias testing."
            )
        
        # All checks passed if no warnings
        is_safe = len(warnings) == 0
        
        return is_safe, warnings
    
    @staticmethod
    def validate_data_privacy(image_path: Optional[Union[str, Path]] = None) -> Tuple[bool, List[str]]:
        """Validate data privacy concerns."""
        warnings = []
        
        if image_path:
            path_str = str(image_path).lower()
            
            # Check for potentially sensitive paths
            sensitive_indicators = [
                "personal", "private", "confidential", "medical", 
                "patient", "id", "passport", "license"
            ]
            
            for indicator in sensitive_indicators:
                if indicator in path_str:
                    warnings.append(
                        f"Image path contains potentially sensitive indicator: {indicator}"
                    )
        
        is_private_safe = len(warnings) == 0
        
        return is_private_safe, warnings