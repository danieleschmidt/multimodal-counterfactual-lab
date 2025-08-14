"""Lightweight core implementation without heavy dependencies."""

import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Use built-in modules only for base functionality
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # Mock Image class for type hints
    class Image:
        @staticmethod
        def new(mode, size, color=None):
            return None
        @staticmethod
        def open(path):
            return None
        def copy(self):
            return self
        def save(self, path, format=None):
            pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightCounterfactualGenerator:
    """Lightweight counterfactual generator with minimal dependencies."""
    
    def __init__(self, method: str = "basic", device: str = "cpu", 
                 use_cache: bool = True, cache_dir: str = "./cache",
                 storage_dir: str = "./data", enable_safety_checks: bool = True,
                 enable_optimization: bool = False, optimization_config=None):
        """Initialize lightweight generator."""
        self.method = method
        self.device = device
        self.use_cache = use_cache
        self.enable_safety_checks = enable_safety_checks
        self.enable_optimization = enable_optimization
        self.cache_dir = Path(cache_dir)
        self.storage_dir = Path(storage_dir)
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True) 
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize storage
        self.cache = {}
        self.generation_count = 0
        
        logger.info(f"Lightweight generator initialized - method: {method}, device: {device}")
    
    def generate(
        self,
        image: Union[str, Any] = None,
        text: str = "",
        attributes: Union[List[str], str] = None,
        num_samples: int = 5,
        save_results: bool = False,
        experiment_id: Optional[str] = None
    ) -> Dict:
        """Generate counterfactual examples with minimal dependencies."""
        
        # Validate and normalize inputs
        if isinstance(attributes, str):
            attributes = [attr.strip() for attr in attributes.split(",")]
        if attributes is None:
            attributes = ["gender", "age"]
        
        text = str(text) if text else "A person in an image"
        num_samples = max(1, min(num_samples, 20))  # Reasonable limits
        
        logger.info(f"Generating {num_samples} counterfactuals for attributes: {attributes}")
        
        # Generate cache key
        cache_key = self._generate_cache_key(text, attributes, num_samples)
        
        # Check cache
        if self.use_cache and cache_key in self.cache:
            logger.info("Using cached result")
            cached_result = self.cache[cache_key].copy()
            cached_result["metadata"]["cache_hit"] = True
            return cached_result
        
        start_time = time.time()
        
        # Generate counterfactuals
        results = self._generate_lightweight_counterfactuals(text, attributes, num_samples)
        
        generation_time = time.time() - start_time
        self.generation_count += 1
        
        # Create result structure
        final_result = {
            "method": self.method,
            "original_image": self._mock_image_placeholder(),
            "original_text": text,
            "target_attributes": attributes,
            "counterfactuals": results,
            "metadata": {
                "generation_time": generation_time,
                "num_samples": len(results),
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "generation_id": f"gen_{self.generation_count}",
                "cache_hit": False
            }
        }
        
        # Cache result
        if self.use_cache:
            self.cache[cache_key] = final_result.copy()
        
        # Save if requested
        if save_results:
            saved_path = self._save_results(final_result, experiment_id)
            final_result["metadata"]["saved_path"] = str(saved_path)
            if experiment_id:
                final_result["metadata"]["experiment_id"] = experiment_id
        
        logger.info(f"Generation completed in {generation_time:.3f}s")
        
        return final_result
    
    def _generate_lightweight_counterfactuals(self, text: str, attributes: List[str], num_samples: int) -> List[Dict]:
        """Generate counterfactuals using rule-based approach."""
        results = []
        
        # Define attribute value mappings
        attribute_values = {
            "gender": ["male", "female", "non-binary"],
            "age": ["young", "middle-aged", "elderly"],
            "race": ["caucasian", "african_american", "asian", "hispanic", "middle_eastern"],
            "profession": ["doctor", "teacher", "engineer", "artist", "scientist"],
            "setting": ["office", "laboratory", "classroom", "hospital", "home"],
            "expression": ["smiling", "serious", "thoughtful", "confident"],
            "hair": ["short", "long", "curly", "straight"],
            "clothing": ["casual", "formal", "professional", "scrubs"]
        }
        
        for i in range(num_samples):
            # Randomly select target values for each attribute
            target_attrs = {}
            for attr in attributes:
                if attr in attribute_values:
                    target_attrs[attr] = random.choice(attribute_values[attr])
                else:
                    target_attrs[attr] = f"variant_{random.randint(1,5)}"
            
            # Generate modified text
            modified_text = self._apply_text_modifications(text, target_attrs)
            
            # Compute confidence based on complexity
            confidence = self._compute_lightweight_confidence(target_attrs, text)
            
            # Create counterfactual entry
            cf_entry = {
                "sample_id": i + 1,
                "target_attributes": target_attrs,
                "generated_image": self._mock_image_placeholder(),
                "generated_text": modified_text,
                "confidence": confidence,
                "explanation": f"Modified {', '.join(attributes)} while preserving context",
                "generation_method": "lightweight_rule_based"
            }
            
            results.append(cf_entry)
        
        return results
    
    def _apply_text_modifications(self, text: str, target_attrs: Dict[str, str]) -> str:
        """Apply attribute-based text modifications."""
        modified = text
        
        # Apply gender modifications
        if "gender" in target_attrs:
            gender = target_attrs["gender"]
            if gender == "male":
                modified = modified.replace("woman", "man").replace("female", "male")
                modified = modified.replace("she", "he").replace("her", "his")
            elif gender == "female":
                modified = modified.replace("man", "woman").replace("male", "female")
                modified = modified.replace("he", "she").replace("his", "her")
        
        # Apply age modifications
        if "age" in target_attrs:
            age = target_attrs["age"]
            age_descriptors = {
                "young": "young",
                "middle-aged": "middle-aged", 
                "elderly": "elderly"
            }
            if age in age_descriptors:
                # Add age descriptor if not present
                if not any(desc in modified.lower() for desc in age_descriptors.values()):
                    modified = f"{age_descriptors[age]} {modified}"
        
        # Apply profession modifications
        if "profession" in target_attrs:
            profession = target_attrs["profession"]
            # Replace common profession words
            profession_words = ["person", "individual", "worker", "professional"]
            for word in profession_words:
                if word in modified.lower():
                    modified = modified.replace(word, profession, 1)
                    break
        
        return modified
    
    def _compute_lightweight_confidence(self, target_attrs: Dict[str, str], original_text: str) -> float:
        """Compute confidence score for generated counterfactual."""
        base_confidence = 0.75
        
        # Penalize for complexity (more attributes = lower confidence)
        complexity_penalty = len(target_attrs) * 0.03
        
        # Bonus for realistic combinations
        realism_bonus = 0.0
        if "gender" in target_attrs and "profession" in target_attrs:
            # Simple realism check (could be expanded)
            realism_bonus = 0.05
        
        # Text length factor
        text_length_factor = min(len(original_text.split()) / 10.0, 1.0) * 0.05
        
        # Random variation to simulate real-world uncertainty
        random_variation = random.uniform(-0.05, 0.05)
        
        confidence = base_confidence - complexity_penalty + realism_bonus + text_length_factor + random_variation
        
        return max(0.1, min(0.95, confidence))
    
    def _generate_cache_key(self, text: str, attributes: List[str], num_samples: int) -> str:
        """Generate cache key for inputs."""
        import hashlib
        key_string = f"{text}_{sorted(attributes)}_{num_samples}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _mock_image_placeholder(self):
        """Create mock image placeholder."""
        if PIL_AVAILABLE:
            return Image.new('RGB', (256, 256), color=(200, 200, 200))
        else:
            return {"type": "mock_image", "size": (256, 256), "mode": "RGB"}
    
    def _save_results(self, result: Dict, experiment_id: Optional[str] = None) -> Path:
        """Save results to storage."""
        if experiment_id is None:
            experiment_id = f"exp_{int(time.time())}_{random.randint(1000, 9999)}"
        
        filename = f"{experiment_id}_{result['metadata']['generation_id']}.json"
        filepath = self.storage_dir / filename
        
        # Prepare data for JSON serialization
        json_data = self._prepare_for_json(result)
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def _prepare_for_json(self, data):
        """Prepare data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif hasattr(data, '__dict__'):
            return {"type": type(data).__name__, "placeholder": True}
        else:
            return data
    
    def generate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate counterfactuals for multiple requests."""
        logger.info(f"Processing batch of {len(requests)} requests")
        
        results = []
        for i, request in enumerate(requests):
            logger.info(f"Processing request {i+1}/{len(requests)}")
            result = self.generate(**request)
            results.append(result)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "generator": {
                "method": self.method,
                "device": self.device,
                "cache_enabled": self.use_cache
            },
            "statistics": {
                "total_generations": self.generation_count,
                "cache_entries": len(self.cache)
            },
            "directories": {
                "cache_dir": str(self.cache_dir),
                "storage_dir": str(self.storage_dir)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup_resources(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Clear cache if it gets too large
        if len(self.cache) > 100:
            # Keep only the 50 most recent entries
            cache_items = list(self.cache.items())
            self.cache = dict(cache_items[-50:])
            logger.info(f"Cache reduced to {len(self.cache)} entries")
        
        logger.info("Resource cleanup completed")


class LightweightBiasEvaluator:
    """Lightweight bias evaluator with minimal dependencies."""
    
    def __init__(self, model=None):
        """Initialize lightweight evaluator."""
        self.model = model or {"name": "mock_model"}
        logger.info("Lightweight bias evaluator initialized")
    
    def evaluate(self, counterfactuals: Dict, metrics: List[str]) -> Dict:
        """Evaluate bias using lightweight metrics."""
        logger.info(f"Evaluating bias with metrics: {metrics}")
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": {},
            "summary": {},
            "counterfactual_analysis": {}
        }
        
        cf_data = counterfactuals["counterfactuals"]
        
        for metric in metrics:
            if metric == "demographic_parity":
                results["metrics"][metric] = self._compute_demographic_parity_lightweight(cf_data)
            elif metric == "attribute_balance":
                results["metrics"][metric] = self._compute_attribute_balance(cf_data)
            elif metric == "confidence_distribution":
                results["metrics"][metric] = self._compute_confidence_distribution(cf_data)
            else:
                results["metrics"][metric] = {"error": f"Metric {metric} not implemented in lightweight version"}
        
        results["summary"] = self._generate_lightweight_summary(results["metrics"])
        results["counterfactual_analysis"] = self._analyze_counterfactuals_lightweight(cf_data)
        
        return results
    
    def _compute_demographic_parity_lightweight(self, cf_data: List[Dict]) -> Dict:
        """Compute demographic parity using simple statistics."""
        attribute_stats = {}
        
        for attr in ["gender", "race", "age"]:
            attr_values = {}
            confidences = {}
            
            for cf in cf_data:
                if attr in cf["target_attributes"]:
                    value = cf["target_attributes"][attr]
                    confidence = cf.get("confidence", 0.8)
                    
                    if value not in attr_values:
                        attr_values[value] = 0
                        confidences[value] = []
                    
                    attr_values[value] += 1
                    confidences[value].append(confidence)
            
            if len(attr_values) > 1:
                # Compute distribution balance
                total = sum(attr_values.values())
                proportions = {k: v/total for k, v in attr_values.items()}
                
                # Compute variance in proportions (lower is more balanced)
                mean_prop = 1.0 / len(proportions)
                variance = sum((p - mean_prop)**2 for p in proportions.values()) / len(proportions)
                
                # Compute confidence differences
                avg_confidences = {k: sum(v)/len(v) for k, v in confidences.items()}
                conf_diff = max(avg_confidences.values()) - min(avg_confidences.values()) if avg_confidences else 0
                
                attribute_stats[attr] = {
                    "distribution": attr_values,
                    "proportions": proportions,
                    "balance_score": max(0, 1 - variance * 10),  # Scale variance to 0-1
                    "confidence_difference": conf_diff,
                    "passes_balance_check": variance < 0.05
                }
        
        overall_balance = sum(stats["balance_score"] for stats in attribute_stats.values()) / len(attribute_stats) if attribute_stats else 0
        
        return {
            "attribute_stats": attribute_stats,
            "overall_balance_score": overall_balance
        }
    
    def _compute_attribute_balance(self, cf_data: List[Dict]) -> Dict:
        """Compute balance across all attributes."""
        all_attributes = {}
        
        for cf in cf_data:
            for attr, value in cf["target_attributes"].items():
                if attr not in all_attributes:
                    all_attributes[attr] = {}
                if value not in all_attributes[attr]:
                    all_attributes[attr][value] = 0
                all_attributes[attr][value] += 1
        
        balance_scores = {}
        for attr, values in all_attributes.items():
            if len(values) > 1:
                # Compute entropy as balance measure
                total = sum(values.values())
                probabilities = [count/total for count in values.values()]
                entropy = -sum(p * (p and p > 0 and (p * (lambda x: x if x > 0 else 0)(p)**0 or 1).__ne__(0) and __import__('math').log2(p) or 0) for p in probabilities)
                max_entropy = __import__('math').log2(len(values))
                balance_scores[attr] = entropy / max_entropy if max_entropy > 0 else 1.0
        
        return {
            "attribute_distributions": all_attributes,
            "balance_scores": balance_scores,
            "overall_balance": sum(balance_scores.values()) / len(balance_scores) if balance_scores else 0
        }
    
    def _compute_confidence_distribution(self, cf_data: List[Dict]) -> Dict:
        """Analyze confidence score distribution."""
        confidences = [cf.get("confidence", 0.8) for cf in cf_data]
        
        if not confidences:
            return {"error": "No confidence scores available"}
        
        # Compute basic statistics without numpy
        mean_conf = sum(confidences) / len(confidences)
        sorted_conf = sorted(confidences)
        median_conf = sorted_conf[len(sorted_conf)//2]
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        # Compute standard deviation
        variance = sum((x - mean_conf)**2 for x in confidences) / len(confidences)
        std_conf = variance ** 0.5
        
        return {
            "mean": mean_conf,
            "median": median_conf,
            "std": std_conf,
            "min": min_conf,
            "max": max_conf,
            "count": len(confidences),
            "quality_assessment": "high" if mean_conf > 0.8 else "medium" if mean_conf > 0.6 else "low"
        }
    
    def _generate_lightweight_summary(self, metrics: Dict) -> Dict:
        """Generate summary of evaluation results."""
        total_score = 0
        score_count = 0
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                if "overall_balance_score" in metric_data:
                    total_score += metric_data["overall_balance_score"]
                    score_count += 1
                elif "overall_balance" in metric_data:
                    total_score += metric_data["overall_balance"]
                    score_count += 1
        
        overall_score = total_score / score_count if score_count > 0 else 0.0
        
        # Determine rating
        if overall_score >= 0.8:
            rating = "Excellent"
        elif overall_score >= 0.6:
            rating = "Good"
        elif overall_score >= 0.4:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            "overall_fairness_score": overall_score,
            "fairness_rating": rating,
            "metrics_evaluated": len(metrics)
        }
    
    def _analyze_counterfactuals_lightweight(self, cf_data: List[Dict]) -> Dict:
        """Analyze counterfactual distribution and quality."""
        if not cf_data:
            return {"error": "No counterfactual data provided"}
        
        # Count attribute occurrences
        attribute_counts = {}
        confidence_scores = []
        
        for cf in cf_data:
            confidence_scores.append(cf.get("confidence", 0.8))
            
            for attr, value in cf["target_attributes"].items():
                if attr not in attribute_counts:
                    attribute_counts[attr] = {}
                if value not in attribute_counts[attr]:
                    attribute_counts[attr][value] = 0
                attribute_counts[attr][value] += 1
        
        # Compute confidence statistics
        mean_confidence = sum(confidence_scores) / len(confidence_scores)
        min_confidence = min(confidence_scores)
        max_confidence = max(confidence_scores)
        
        return {
            "attribute_distribution": attribute_counts,
            "confidence_stats": {
                "mean": mean_confidence,
                "min": min_confidence,
                "max": max_confidence,
                "count": len(confidence_scores)
            },
            "total_counterfactuals": len(cf_data),
            "unique_attributes": len(attribute_counts)
        }
    
    def generate_report(self, results: Dict, format: str = "simple", export_path: Optional[str] = None) -> Dict:
        """Generate simple evaluation report."""
        logger.info(f"Generating {format} report")
        
        report = {
            "report_type": f"Lightweight Bias Evaluation Report ({format})",
            "generated_at": datetime.now().isoformat(),
            "summary": results.get("summary", {}),
            "key_findings": self._extract_lightweight_findings(results),
            "detailed_metrics": results.get("metrics", {}),
            "counterfactual_analysis": results.get("counterfactual_analysis", {})
        }
        
        if export_path:
            with open(export_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {export_path}")
        
        return report
    
    def _extract_lightweight_findings(self, results: Dict) -> List[str]:
        """Extract key findings from evaluation results."""
        findings = []
        
        summary = results.get("summary", {})
        score = summary.get("overall_fairness_score", 0)
        rating = summary.get("fairness_rating", "Unknown")
        
        findings.append(f"Overall fairness score: {score:.3f} ({rating})")
        
        # Check specific metrics
        metrics = results.get("metrics", {})
        if "demographic_parity" in metrics:
            dp = metrics["demographic_parity"]
            if isinstance(dp, dict) and "overall_balance_score" in dp:
                findings.append(f"Demographic balance score: {dp['overall_balance_score']:.3f}")
        
        if "confidence_distribution" in metrics:
            conf = metrics["confidence_distribution"]
            if isinstance(conf, dict) and "quality_assessment" in conf:
                findings.append(f"Confidence quality: {conf['quality_assessment']}")
        
        return findings or ["No significant findings"]