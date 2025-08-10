"""Minimal working core without external dependencies for initial testing."""

import json
import logging
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockImage:
    """Mock image class for dependency-free testing."""
    
    def __init__(self, width: int = 400, height: int = 300, mode: str = "RGB"):
        self.width = width
        self.height = height
        self.mode = mode
        self.format = None
        self.data = f"mock_image_{width}x{height}_{random.randint(1000, 9999)}"
    
    def copy(self):
        """Return a copy of this mock image."""
        return MockImage(self.width, self.height, self.mode)
    
    def save(self, fp, format=None):
        """Mock save operation."""
        if hasattr(fp, 'write'):
            fp.write(f"MOCK_IMAGE_DATA_{self.data}")
        else:
            with open(fp, 'w') as f:
                f.write(f"MOCK_IMAGE_DATA_{self.data}")
    
    def __str__(self):
        return f"MockImage({self.width}x{self.height}, {self.mode})"


class MinimalCounterfactualGenerator:
    """Minimal counterfactual generator without external dependencies."""
    
    def __init__(self, method: str = "modicf", device: str = "cpu"):
        """Initialize minimal generator."""
        self.method = method
        self.device = device
        self.generation_count = 0
        
        logger.info(f"Minimal generator initialized: method={method}, device={device}")
    
    def generate(
        self,
        image: Union[str, Path, MockImage],
        text: str,
        attributes: Union[List[str], str],
        num_samples: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate counterfactual examples."""
        
        self.generation_count += 1
        
        # Validate inputs
        if isinstance(image, str):
            # In real implementation, would load image
            mock_image = MockImage(400, 300)
        elif isinstance(image, Path):
            mock_image = MockImage(400, 300)
        else:
            mock_image = image
        
        if isinstance(attributes, str):
            attributes = [attr.strip() for attr in attributes.split(',')]
        
        logger.info(f"Generating {num_samples} counterfactuals for attributes: {attributes}")
        
        start_time = datetime.now()
        
        # Generate counterfactuals
        results = []
        attribute_values = {
            "gender": ["male", "female", "non-binary"],
            "race": ["white", "black", "asian", "hispanic", "diverse"],
            "age": ["young", "middle-aged", "elderly"]
        }
        
        for i in range(num_samples):
            target_attrs = {}
            for attr in attributes:
                if attr in attribute_values:
                    target_attrs[attr] = random.choice(attribute_values[attr])
            
            # Generate mock counterfactual
            generated_image = mock_image.copy()
            
            # Mock text modification
            modified_text = self._modify_text(text, target_attrs)
            
            results.append({
                "sample_id": i,
                "target_attributes": target_attrs,
                "generated_image": generated_image,
                "generated_text": modified_text,
                "confidence": random.uniform(0.7, 0.95),
                "explanation": f"Applied {self.method} to modify {', '.join(attributes)}"
            })
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "method": self.method,
            "original_image": mock_image,
            "original_text": text,
            "target_attributes": attributes,
            "counterfactuals": results,
            "metadata": {
                "generation_time": generation_time,
                "num_samples": len(results),
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "generation_id": self.generation_count
            }
        }
    
    def _modify_text(self, text: str, target_attrs: Dict[str, str]) -> str:
        """Modify text based on target attributes."""
        modified = text
        
        for attr, value in target_attrs.items():
            if attr == "gender":
                if "man" in modified.lower():
                    if value == "female":
                        modified = modified.replace("man", "woman").replace("Man", "Woman")
                elif "woman" in modified.lower():
                    if value == "male":
                        modified = modified.replace("woman", "man").replace("Woman", "Man")
            elif attr == "age":
                if value == "elderly":
                    modified = f"elderly {modified}" if not any(age in modified.lower() for age in ["young", "old", "elderly"]) else modified
                elif value == "young":
                    modified = f"young {modified}" if not any(age in modified.lower() for age in ["young", "old", "elderly"]) else modified
        
        return modified
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "method": self.method,
            "device": self.device,
            "generations_completed": self.generation_count,
            "status": "operational",
            "timestamp": datetime.now().isoformat()
        }


class MinimalBiasEvaluator:
    """Minimal bias evaluator without external dependencies."""
    
    def __init__(self):
        """Initialize minimal evaluator."""
        self.evaluation_count = 0
        logger.info("Minimal bias evaluator initialized")
    
    def evaluate(
        self,
        counterfactuals: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Evaluate bias using specified metrics."""
        
        self.evaluation_count += 1
        
        logger.info(f"Evaluating bias with metrics: {metrics}")
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_id": self.evaluation_count,
            "metrics": {},
            "summary": {},
            "counterfactual_analysis": {}
        }
        
        cf_data = counterfactuals["counterfactuals"]
        
        for metric in metrics:
            if metric == "demographic_parity":
                results["metrics"][metric] = self._compute_demographic_parity(cf_data)
            elif metric == "equalized_odds":
                results["metrics"][metric] = self._compute_equalized_odds(cf_data)
            elif metric == "cits_score":
                results["metrics"][metric] = self._compute_cits_score(cf_data)
            elif metric == "fairness_score":
                results["metrics"][metric] = self._compute_fairness_score(cf_data)
            else:
                results["metrics"][metric] = {"error": f"Unknown metric: {metric}"}
        
        results["summary"] = self._generate_summary(results["metrics"])
        results["counterfactual_analysis"] = self._analyze_counterfactuals(cf_data)
        
        return results
    
    def _compute_demographic_parity(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Mock demographic parity computation."""
        # Simulate demographic parity violation detection
        violations = 0
        total_groups = 0
        
        for attr in ["gender", "race", "age"]:
            groups = {}
            for cf in cf_data:
                if attr in cf["target_attributes"]:
                    value = cf["target_attributes"][attr]
                    if value not in groups:
                        groups[value] = []
                    groups[value].append(cf["confidence"])
            
            if len(groups) > 1:
                total_groups += len(groups)
                means = [sum(scores)/len(scores) for scores in groups.values()]
                max_diff = max(means) - min(means)
                if max_diff > 0.1:  # Fairness threshold
                    violations += 1
        
        parity_score = 1 - (violations / max(total_groups, 1))
        
        return {
            "parity_score": parity_score,
            "violations_detected": violations,
            "total_groups": total_groups,
            "passes_threshold": parity_score >= 0.8
        }
    
    def _compute_equalized_odds(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Mock equalized odds computation."""
        # Simulate equalized odds analysis
        tpr_differences = []
        
        for cf in cf_data:
            # Mock true positive rate difference
            base_tpr = random.uniform(0.7, 0.9)
            cf_tpr = random.uniform(0.6, 0.95)
            tpr_differences.append(abs(base_tpr - cf_tpr))
        
        avg_tpr_diff = sum(tpr_differences) / len(tpr_differences) if tpr_differences else 0
        
        return {
            "avg_tpr_difference": avg_tpr_diff,
            "max_tpr_difference": max(tpr_differences) if tpr_differences else 0,
            "passes_threshold": avg_tpr_diff < 0.1
        }
    
    def _compute_cits_score(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Mock CITS score computation."""
        scores = []
        
        for cf in cf_data:
            # Mock CITS calculation based on confidence and attribute complexity
            confidence = cf["confidence"]
            num_attributes = len(cf["target_attributes"])
            
            # Higher score for high confidence and fewer attribute changes
            cits = confidence * (1 - (num_attributes - 1) * 0.1)
            scores.append(max(0, min(1, cits)))
        
        return {
            "individual_scores": scores,
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "std_score": self._compute_std(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0
        }
    
    def _compute_fairness_score(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Compute overall fairness score."""
        # Aggregate fairness metrics
        total_confidence = sum(cf["confidence"] for cf in cf_data)
        avg_confidence = total_confidence / len(cf_data) if cf_data else 0
        
        # Penalize for attribute imbalance
        attribute_counts = {}
        for cf in cf_data:
            for attr, value in cf["target_attributes"].items():
                key = f"{attr}_{value}"
                attribute_counts[key] = attribute_counts.get(key, 0) + 1
        
        # Calculate distribution balance
        if attribute_counts:
            counts = list(attribute_counts.values())
            balance_score = 1 - (max(counts) - min(counts)) / max(counts)
        else:
            balance_score = 1.0
        
        overall_score = (avg_confidence + balance_score) / 2
        
        return {
            "overall_fairness_score": overall_score,
            "confidence_component": avg_confidence,
            "balance_component": balance_score,
            "attribute_distribution": attribute_counts,
            "rating": self._get_fairness_rating(overall_score)
        }
    
    def _compute_std(self, scores: List[float]) -> float:
        """Compute standard deviation."""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation summary."""
        passed_metrics = 0
        total_metrics = 0
        
        for metric_data in metrics.values():
            if isinstance(metric_data, dict) and "passes_threshold" in metric_data:
                total_metrics += 1
                if metric_data["passes_threshold"]:
                    passed_metrics += 1
        
        fairness_score = passed_metrics / total_metrics if total_metrics > 0 else 0.0
        
        return {
            "overall_fairness_score": fairness_score,
            "passed_metrics": passed_metrics,
            "total_metrics": total_metrics,
            "fairness_rating": self._get_fairness_rating(fairness_score)
        }
    
    def _get_fairness_rating(self, score: float) -> str:
        """Convert score to rating."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def _analyze_counterfactuals(self, cf_data: List[Dict]) -> Dict[str, Any]:
        """Analyze counterfactual distribution."""
        attribute_dist = {}
        confidence_scores = []
        
        for cf in cf_data:
            confidence_scores.append(cf["confidence"])
            
            for attr, value in cf["target_attributes"].items():
                if attr not in attribute_dist:
                    attribute_dist[attr] = {}
                if value not in attribute_dist[attr]:
                    attribute_dist[attr][value] = 0
                attribute_dist[attr][value] += 1
        
        return {
            "attribute_distribution": attribute_dist,
            "confidence_stats": {
                "mean": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0,
                "std": self._compute_std(confidence_scores)
            },
            "total_counterfactuals": len(cf_data)
        }
    
    def generate_report(
        self,
        results: Dict[str, Any],
        format: str = "regulatory",
        export_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate bias audit report."""
        logger.info(f"Generating {format} report")
        
        report = {
            "report_type": f"AI Bias Audit Report ({format.title()})",
            "report_date": datetime.now().isoformat(),
            "evaluation_id": results.get("evaluation_id", "unknown"),
            "executive_summary": {
                "overall_assessment": results["summary"]["fairness_rating"],
                "fairness_score": results["summary"]["overall_fairness_score"],
                "total_counterfactuals": results["counterfactual_analysis"]["total_counterfactuals"],
                "metrics_evaluated": list(results["metrics"].keys())
            },
            "detailed_findings": results["metrics"],
            "recommendations": self._generate_recommendations(results),
            "methodology": {
                "evaluation_approach": "Minimal counterfactual fairness evaluation",
                "sample_size": results["counterfactual_analysis"]["total_counterfactuals"]
            }
        }
        
        if export_path:
            with open(export_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report exported to {export_path}")
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        score = results["summary"]["overall_fairness_score"]
        
        if score < 0.7:
            recommendations.extend([
                "Implement bias mitigation strategies",
                "Increase diversity in training data",
                "Regular bias monitoring and evaluation"
            ])
        elif score < 0.9:
            recommendations.extend([
                "Continue current practices with minor improvements",
                "Monitor edge cases more closely"
            ])
        else:
            recommendations.append("Maintain current excellent fairness practices")
        
        return recommendations


# Simple test function to verify functionality
def test_minimal_system():
    """Test the minimal system functionality."""
    logger.info("Testing minimal counterfactual lab system...")
    
    # Test generator
    generator = MinimalCounterfactualGenerator(method="modicf")
    
    # Create test image
    test_image = MockImage(400, 300)
    test_text = "A doctor examining a patient"
    test_attributes = ["gender", "age"]
    
    # Generate counterfactuals
    results = generator.generate(
        image=test_image,
        text=test_text,
        attributes=test_attributes,
        num_samples=3
    )
    
    logger.info(f"Generated {len(results['counterfactuals'])} counterfactuals")
    
    # Test evaluator
    evaluator = MinimalBiasEvaluator()
    
    # Evaluate bias
    evaluation = evaluator.evaluate(
        counterfactuals=results,
        metrics=["demographic_parity", "cits_score", "fairness_score"]
    )
    
    logger.info(f"Evaluation completed with fairness rating: {evaluation['summary']['fairness_rating']}")
    
    # Generate report
    report = evaluator.generate_report(evaluation, format="technical")
    
    logger.info("✅ Minimal system test completed successfully!")
    
    return {
        "generation_results": results,
        "evaluation_results": evaluation,
        "audit_report": report
    }


if __name__ == "__main__":
    test_minimal_system()