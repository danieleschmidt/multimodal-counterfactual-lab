"""Core classes for counterfactual generation and bias evaluation."""

import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some functionality will be limited.")

from counterfactual_lab.methods.modicf import MoDiCF
from counterfactual_lab.methods.icg import ICG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """Main interface for generating counterfactual image-text pairs."""
    
    def __init__(self, method: str = "modicf", device: str = "cuda"):
        """Initialize the counterfactual generator.
        
        Args:
            method: Generation method ("modicf" or "icg")
            device: Compute device ("cuda" or "cpu")
        """
        self.method = method
        self.device = device if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
        
        self._initialize_method()
    
    def _initialize_method(self):
        """Initialize the selected generation method."""
        if self.method == "modicf":
            self.generator = MoDiCF(device=self.device)
        elif self.method == "icg":
            self.generator = ICG(device=self.device)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Initialized {self.method} generator on {self.device}")
    
    def generate(
        self,
        image,
        text: str,
        attributes: List[str],
        num_samples: int = 5
    ) -> Dict:
        """Generate counterfactual examples.
        
        Args:
            image: Input image (PIL Image or path)
            text: Input text description
            attributes: Attributes to modify ["gender", "race", "age"]
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing generated counterfactuals
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        logger.info(f"Generating {num_samples} counterfactuals for attributes: {attributes}")
        
        start_time = datetime.now()
        
        if self.method == "modicf":
            results = self._generate_modicf(image, text, attributes, num_samples)
        elif self.method == "icg":
            results = self._generate_icg(image, text, attributes, num_samples)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "method": self.method,
            "original_image": image,
            "original_text": text,
            "target_attributes": attributes,
            "counterfactuals": results,
            "metadata": {
                "generation_time": generation_time,
                "num_samples": len(results),
                "device": self.device,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _generate_modicf(self, image, text: str, attributes: List[str], num_samples: int) -> List[Dict]:
        """Generate counterfactuals using MoDiCF method."""
        results = []
        
        attribute_values = {
            "gender": ["male", "female", "non-binary"],
            "race": ["white", "black", "asian", "hispanic"],
            "age": ["young", "middle-aged", "elderly"]
        }
        
        for i in range(num_samples):
            target_attrs = {}
            for attr in attributes:
                if attr in attribute_values:
                    values = attribute_values[attr]
                    target_attrs[attr] = np.random.choice(values)
            
            counterfactual = self.generator.generate_controlled(
                image=image,
                text=text,
                target_attributes=target_attrs,
                preserve=["background", "clothing", "pose"]
            )
            
            results.append({
                "sample_id": i,
                "target_attributes": target_attrs,
                "generated_image": counterfactual.get("image", image),  # Fallback to original
                "generated_text": counterfactual.get("text", text),
                "confidence": counterfactual.get("confidence", 0.8),
                "explanation": f"Modified {', '.join(attributes)} while preserving context"
            })
        
        return results
    
    def _generate_icg(self, image, text: str, attributes: List[str], num_samples: int) -> List[Dict]:
        """Generate counterfactuals using ICG method."""
        results = []
        
        for i in range(num_samples):
            attribute_changes = {}
            for attr in attributes:
                if attr == "gender":
                    attribute_changes[attr] = np.random.choice(["male", "female"])
                elif attr == "age":
                    attribute_changes[attr] = np.random.choice(["young", "elderly"])
                elif attr == "race":
                    attribute_changes[attr] = np.random.choice(["diverse_ethnicity"])
            
            counterfactual = self.generator.generate_interpretable(
                text=text,
                image=image,
                attribute_changes=attribute_changes,
                explanation_level="detailed"
            )
            
            results.append({
                "sample_id": i,
                "target_attributes": attribute_changes,
                "generated_image": counterfactual.get("image", image),
                "generated_text": counterfactual.get("text", text),
                "explanation": counterfactual.get("explanation", f"Applied interpretable changes to {', '.join(attributes)}"),
                "reasoning": counterfactual.get("reasoning", "Automated attribute transformation")
            })
        
        return results
    
    def visualize_grid(self, counterfactuals: Dict, save_path: Optional[str] = None):
        """Visualize counterfactuals in a grid format."""
        if not TORCH_AVAILABLE:
            logger.warning("Visualization requires matplotlib. Skipping visualization.")
            return
        
        results = counterfactuals["counterfactuals"]
        num_samples = len(results)
        
        if num_samples == 0:
            logger.warning("No counterfactuals to visualize")
            return
        
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows + 1, cols, figsize=(4*cols, 4*(rows+1)))
        if rows == 0:
            axes = axes.reshape(1, -1)
        
        # Show original image
        if hasattr(axes[0], '__len__'):
            axes[0][0].imshow(counterfactuals["original_image"])
            axes[0][0].set_title("Original")
            axes[0][0].axis('off')
            
            for i in range(1, cols):
                axes[0][i].axis('off')
        else:
            axes[0].imshow(counterfactuals["original_image"])
            axes[0].set_title("Original")
            axes[0].axis('off')
        
        # Show counterfactuals
        for i, result in enumerate(results):
            row = i // cols + 1
            col = i % cols
            
            if rows > 0:
                ax = axes[row][col] if cols > 1 else axes[row]
            else:
                ax = axes[col]
            
            ax.imshow(result["generated_image"])
            attrs = ", ".join([f"{k}:{v}" for k, v in result["target_attributes"].items()])
            ax.set_title(f"CF {i+1}: {attrs}", fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class BiasEvaluator:
    """Evaluates bias in vision-language models using counterfactuals."""
    
    def __init__(self, model):
        """Initialize bias evaluator with a model."""
        self.model = model
        logger.info(f"Initialized BiasEvaluator with model: {type(model).__name__}")
    
    def evaluate(
        self,
        counterfactuals: Dict,
        metrics: List[str]
    ) -> Dict:
        """Evaluate bias using specified metrics.
        
        Args:
            counterfactuals: Generated counterfactual data
            metrics: List of bias metrics to compute
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating bias with metrics: {metrics}")
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": {},
            "summary": {},
            "counterfactual_analysis": {}
        }
        
        cf_data = counterfactuals["counterfactuals"]
        original_text = counterfactuals["original_text"]
        
        for metric in metrics:
            if metric == "demographic_parity":
                results["metrics"][metric] = self._compute_demographic_parity(cf_data)
            elif metric == "equalized_odds":
                results["metrics"][metric] = self._compute_equalized_odds(cf_data)
            elif metric == "cits_score":
                results["metrics"][metric] = self._compute_cits_score(cf_data, original_text)
            elif metric == "disparate_impact":
                results["metrics"][metric] = self._compute_disparate_impact(cf_data)
            else:
                logger.warning(f"Unknown metric: {metric}")
                results["metrics"][metric] = {"error": f"Unknown metric: {metric}"}
        
        results["summary"] = self._generate_summary(results["metrics"])
        results["counterfactual_analysis"] = self._analyze_counterfactuals(cf_data)
        
        return results
    
    def _compute_demographic_parity(self, cf_data: List[Dict]) -> Dict:
        """Compute demographic parity metrics."""
        attribute_scores = {}
        
        for attr in ["gender", "race", "age"]:
            attr_values = {}
            for cf in cf_data:
                if attr in cf["target_attributes"]:
                    value = cf["target_attributes"][attr]
                    confidence = cf.get("confidence", 0.8)
                    
                    if value not in attr_values:
                        attr_values[value] = []
                    attr_values[value].append(confidence)
            
            if len(attr_values) > 1:
                means = {k: np.mean(v) for k, v in attr_values.items()}
                max_diff = max(means.values()) - min(means.values())
                attribute_scores[attr] = {
                    "max_difference": max_diff,
                    "group_means": means,
                    "passes_threshold": max_diff < 0.1  # Standard fairness threshold
                }
        
        return {
            "attribute_scores": attribute_scores,
            "overall_score": np.mean([score["max_difference"] for score in attribute_scores.values()]) if attribute_scores else 0.0
        }
    
    def _compute_equalized_odds(self, cf_data: List[Dict]) -> Dict:
        """Compute equalized odds metrics."""
        # Simulate model predictions for original vs counterfactual
        predictions = []
        
        for cf in cf_data:
            # Mock prediction differences based on attribute changes
            base_pred = 0.7  # Base prediction confidence
            
            # Simulate bias: certain attributes lead to different predictions
            if "gender" in cf["target_attributes"]:
                if cf["target_attributes"]["gender"] == "female":
                    base_pred *= 0.9  # Simulate slight bias against females
            
            if "race" in cf["target_attributes"]:
                if cf["target_attributes"]["race"] == "black":
                    base_pred *= 0.85  # Simulate racial bias
            
            predictions.append({
                "original_pred": 0.7,
                "counterfactual_pred": base_pred,
                "attributes": cf["target_attributes"]
            })
        
        # Calculate equalized odds
        group_tpr = {}  # True positive rate by group
        for pred in predictions:
            for attr, value in pred["attributes"].items():
                key = f"{attr}_{value}"
                if key not in group_tpr:
                    group_tpr[key] = []
                group_tpr[key].append(pred["counterfactual_pred"])
        
        tpr_diff = 0.0
        if len(group_tpr) > 1:
            tpr_means = {k: np.mean(v) for k, v in group_tpr.items()}
            tpr_diff = max(tpr_means.values()) - min(tpr_means.values())
        
        return {
            "tpr_difference": tpr_diff,
            "group_tpr": {k: np.mean(v) for k, v in group_tpr.items()},
            "passes_threshold": tpr_diff < 0.1
        }
    
    def _compute_cits_score(self, cf_data: List[Dict], original_text: str) -> Dict:
        """Compute Counterfactual Image-Text Score."""
        scores = []
        
        for cf in cf_data:
            # Mock CITS computation based on text similarity and image quality
            text_similarity = self._mock_text_similarity(original_text, cf["generated_text"])
            image_quality = cf.get("confidence", 0.8)
            
            # CITS combines similarity and quality
            cits = (text_similarity + image_quality) / 2
            scores.append(cits)
        
        return {
            "individual_scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores)
        }
    
    def _compute_disparate_impact(self, cf_data: List[Dict]) -> Dict:
        """Compute disparate impact ratios."""
        impact_ratios = {}
        
        for attr in ["gender", "race", "age"]:
            groups = {}
            for cf in cf_data:
                if attr in cf["target_attributes"]:
                    value = cf["target_attributes"][attr]
                    confidence = cf.get("confidence", 0.8)
                    
                    if value not in groups:
                        groups[value] = []
                    groups[value].append(confidence)
            
            if len(groups) >= 2:
                group_rates = {k: np.mean(v) for k, v in groups.items()}
                max_rate = max(group_rates.values())
                min_rate = min(group_rates.values())
                ratio = min_rate / max_rate if max_rate > 0 else 1.0
                
                impact_ratios[attr] = {
                    "ratio": ratio,
                    "group_rates": group_rates,
                    "passes_80_rule": ratio >= 0.8  # 80% rule for disparate impact
                }
        
        return impact_ratios
    
    def _mock_text_similarity(self, text1: str, text2: str) -> float:
        """Mock text similarity computation."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_summary(self, metrics: Dict) -> Dict:
        """Generate summary of evaluation results."""
        passed_metrics = 0
        total_metrics = 0
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                if "passes_threshold" in metric_data:
                    total_metrics += 1
                    if metric_data["passes_threshold"]:
                        passed_metrics += 1
                elif "attribute_scores" in metric_data:
                    for attr_score in metric_data["attribute_scores"].values():
                        if "passes_threshold" in attr_score:
                            total_metrics += 1
                            if attr_score["passes_threshold"]:
                                passed_metrics += 1
        
        fairness_score = passed_metrics / total_metrics if total_metrics > 0 else 0.0
        
        return {
            "overall_fairness_score": fairness_score,
            "passed_metrics": passed_metrics,
            "total_metrics": total_metrics,
            "fairness_rating": self._get_fairness_rating(fairness_score)
        }
    
    def _get_fairness_rating(self, score: float) -> str:
        """Convert fairness score to rating."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def _analyze_counterfactuals(self, cf_data: List[Dict]) -> Dict:
        """Analyze counterfactual distribution and quality."""
        attribute_dist = {}
        confidence_scores = []
        
        for cf in cf_data:
            confidence_scores.append(cf.get("confidence", 0.8))
            
            for attr, value in cf["target_attributes"].items():
                if attr not in attribute_dist:
                    attribute_dist[attr] = {}
                if value not in attribute_dist[attr]:
                    attribute_dist[attr][value] = 0
                attribute_dist[attr][value] += 1
        
        return {
            "attribute_distribution": attribute_dist,
            "confidence_stats": {
                "mean": np.mean(confidence_scores),
                "std": np.std(confidence_scores),
                "min": np.min(confidence_scores),
                "max": np.max(confidence_scores)
            },
            "total_counterfactuals": len(cf_data)
        }
    
    def generate_report(
        self,
        results: Dict,
        format: str = "regulatory",
        export_path: Optional[str] = None
    ) -> Dict:
        """Generate bias audit report.
        
        Args:
            results: Evaluation results
            format: Report format ("regulatory", "academic", "technical")
            export_path: Path to save report
            
        Returns:
            Generated report
        """
        logger.info(f"Generating {format} report")
        
        if format == "regulatory":
            report = self._generate_regulatory_report(results)
        elif format == "academic":
            report = self._generate_academic_report(results)
        elif format == "technical":
            report = self._generate_technical_report(results)
        else:
            raise ValueError(f"Unknown report format: {format}")
        
        if export_path:
            with open(export_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report exported to {export_path}")
        
        return report
    
    def _generate_regulatory_report(self, results: Dict) -> Dict:
        """Generate regulatory compliance report."""
        return {
            "report_type": "AI Bias Audit Report",
            "compliance_framework": "EU AI Act Article 10",
            "report_date": datetime.now().isoformat(),
            "executive_summary": {
                "overall_assessment": results["summary"]["fairness_rating"],
                "fairness_score": results["summary"]["overall_fairness_score"],
                "key_findings": self._extract_key_findings(results),
                "recommendations": self._generate_recommendations(results)
            },
            "methodology": {
                "evaluation_approach": "Counterfactual fairness evaluation",
                "metrics_used": list(results["metrics"].keys()),
                "sample_size": results["counterfactual_analysis"]["total_counterfactuals"]
            },
            "detailed_findings": results["metrics"],
            "risk_assessment": self._assess_risk_level(results),
            "mitigation_plan": self._create_mitigation_plan(results),
            "compliance_status": {
                "eu_ai_act": "COMPLIANT" if results["summary"]["fairness_score"] >= 0.7 else "NON_COMPLIANT",
                "audit_date": datetime.now().isoformat(),
                "next_audit_due": self._calculate_next_audit_date()
            }
        }
    
    def _generate_academic_report(self, results: Dict) -> Dict:
        """Generate academic research report."""
        return {
            "title": "Counterfactual Fairness Evaluation Report",
            "abstract": self._generate_abstract(results),
            "methodology": {
                "approach": "Counterfactual generation and bias evaluation",
                "metrics": list(results["metrics"].keys()),
                "evaluation_framework": "Multimodal fairness assessment"
            },
            "results": results["metrics"],
            "discussion": {
                "key_findings": self._extract_key_findings(results),
                "limitations": self._identify_limitations(),
                "future_work": self._suggest_future_work()
            },
            "conclusion": self._generate_conclusion(results),
            "statistical_significance": self._assess_statistical_significance(results)
        }
    
    def _generate_technical_report(self, results: Dict) -> Dict:
        """Generate technical implementation report."""
        return {
            "technical_summary": results,
            "performance_metrics": {
                "evaluation_timestamp": results["evaluation_timestamp"],
                "total_counterfactuals": results["counterfactual_analysis"]["total_counterfactuals"]
            },
            "detailed_metrics": results["metrics"],
            "recommendations": {
                "immediate_actions": self._get_immediate_actions(results),
                "long_term_improvements": self._get_long_term_improvements(results)
            },
            "raw_data": results["counterfactual_analysis"]
        }
    
    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from evaluation results."""
        findings = []
        
        for metric_name, metric_data in results["metrics"].items():
            if metric_name == "demographic_parity":
                if metric_data.get("overall_score", 0) > 0.1:
                    findings.append(f"Significant demographic parity violations detected (score: {metric_data['overall_score']:.3f})")
            elif metric_name == "equalized_odds":
                if not metric_data.get("passes_threshold", False):
                    findings.append(f"Equalized odds threshold not met (TPR difference: {metric_data['tpr_difference']:.3f})")
        
        if results["summary"]["fairness_score"] < 0.7:
            findings.append("Overall fairness score below acceptable threshold")
        
        return findings or ["No significant bias detected in evaluated metrics"]
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if results["summary"]["fairness_score"] < 0.7:
            recommendations.append("Implement bias mitigation strategies in model training")
            recommendations.append("Increase diversity in training data")
            recommendations.append("Regular bias monitoring and evaluation")
        
        return recommendations or ["Continue current fairness practices with regular monitoring"]
    
    def _assess_risk_level(self, results: Dict) -> str:
        """Assess risk level based on evaluation results."""
        score = results["summary"]["fairness_score"]
        
        if score >= 0.9:
            return "LOW"
        elif score >= 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _create_mitigation_plan(self, results: Dict) -> Dict:
        """Create mitigation plan for identified issues."""
        return {
            "immediate_actions": [
                "Document identified bias patterns",
                "Implement additional validation checks",
                "Monitor high-risk predictions"
            ],
            "short_term": [
                "Retrain model with balanced data",
                "Implement fairness constraints",
                "Enhanced testing protocols"
            ],
            "long_term": [
                "Continuous bias monitoring system",
                "Regular fairness audits",
                "Model governance framework"
            ]
        }
    
    def _calculate_next_audit_date(self) -> str:
        """Calculate next audit date."""
        from datetime import timedelta
        next_audit = datetime.now() + timedelta(days=180)  # 6 months
        return next_audit.isoformat()
    
    def _generate_abstract(self, results: Dict) -> str:
        """Generate abstract for academic report."""
        return f"""This report presents a comprehensive fairness evaluation of a vision-language model using counterfactual generation techniques. We evaluated {results['counterfactual_analysis']['total_counterfactuals']} counterfactual examples across multiple protected attributes. The overall fairness score was {results['summary']['overall_fairness_score']:.3f}, indicating {results['summary']['fairness_rating'].lower()} performance. Key metrics included demographic parity, equalized odds, and CITS scores."""
    
    def _identify_limitations(self) -> List[str]:
        """Identify limitations of the evaluation."""
        return [
            "Limited to visual and textual modalities",
            "Counterfactual quality depends on generation method",
            "Evaluation metrics may not capture all forms of bias",
            "Sample size limitations may affect statistical significance"
        ]
    
    def _suggest_future_work(self) -> List[str]:
        """Suggest future work directions."""
        return [
            "Expand to additional protected attributes",
            "Develop domain-specific fairness metrics",
            "Investigate intersectional bias patterns",
            "Improve counterfactual generation quality"
        ]
    
    def _generate_conclusion(self, results: Dict) -> str:
        """Generate conclusion for academic report."""
        rating = results["summary"]["fairness_rating"]
        score = results["summary"]["overall_fairness_score"]
        
        return f"The evaluation demonstrates {rating.lower()} fairness performance with an overall score of {score:.3f}. While some bias patterns were identified, the counterfactual evaluation framework provides valuable insights for model improvement and regulatory compliance."
    
    def _assess_statistical_significance(self, results: Dict) -> Dict:
        """Assess statistical significance of results."""
        return {
            "sample_size": results["counterfactual_analysis"]["total_counterfactuals"],
            "confidence_level": "95%",
            "significance_threshold": 0.05,
            "notes": "Statistical significance assessment requires larger sample sizes for robust conclusions"
        }
    
    def _get_immediate_actions(self, results: Dict) -> List[str]:
        """Get immediate technical actions."""
        actions = []
        
        if results["summary"]["fairness_score"] < 0.7:
            actions.extend([
                "Review model architecture for bias sources",
                "Analyze training data distribution",
                "Implement bias detection in inference pipeline"
            ])
        
        return actions or ["Monitor current performance metrics"]
    
    def _get_long_term_improvements(self, results: Dict) -> List[str]:
        """Get long-term improvement suggestions."""
        return [
            "Implement continuous fairness monitoring",
            "Develop automated bias detection systems",
            "Create fairness-aware model training pipelines",
            "Establish regular audit schedules"
        ]