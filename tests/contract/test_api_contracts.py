"""Contract testing for API consistency and backward compatibility."""

import pytest
import json
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional
from counterfactual_lab import CounterfactualGenerator, BiasEvaluator


class CounterfactualRequest(BaseModel):
    """Contract for counterfactual generation requests."""
    image_path: str
    text: str
    attributes: List[str]
    num_samples: int = 5
    method: str = "modicf"
    
    class Config:
        extra = "forbid"  # Strict contract - no additional fields allowed


class CounterfactualResponse(BaseModel):
    """Contract for counterfactual generation responses."""
    counterfactuals: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    generation_time: float
    method_used: str
    
    class Config:
        extra = "forbid"


class BiasEvaluationRequest(BaseModel):
    """Contract for bias evaluation requests."""
    counterfactuals: List[Dict[str, Any]]
    metrics: List[str]
    protected_attributes: List[str]
    
    class Config:
        extra = "forbid"


class BiasEvaluationResponse(BaseModel):
    """Contract for bias evaluation responses."""
    bias_scores: Dict[str, float]
    fairness_metrics: Dict[str, float]
    evaluation_time: float
    recommendations: List[str]
    
    class Config:
        extra = "forbid"


class TestAPIContracts:
    """Test API contracts for backward compatibility and consistency."""
    
    def test_counterfactual_generation_contract(self):
        """Test counterfactual generation API contract adherence."""
        generator = CounterfactualGenerator(method="modicf")
        
        # Valid request
        request_data = {
            "image_path": "test_image.jpg",
            "text": "A person walking",
            "attributes": ["gender", "age"],
            "num_samples": 3,
            "method": "modicf"
        }
        
        # Validate request contract
        request = CounterfactualRequest(**request_data)
        assert request.image_path == "test_image.jpg"
        assert request.attributes == ["gender", "age"]
        
        # Mock response validation
        mock_response = {
            "counterfactuals": [
                {"image": "path1.jpg", "text": "modified text 1"},
                {"image": "path2.jpg", "text": "modified text 2"}
            ],
            "metadata": {"model_version": "1.0", "timestamp": "2025-01-01"},
            "generation_time": 2.5,
            "method_used": "modicf"
        }
        
        response = CounterfactualResponse(**mock_response)
        assert len(response.counterfactuals) == 2
        assert response.method_used == "modicf"
    
    def test_bias_evaluation_contract(self):
        """Test bias evaluation API contract adherence."""
        evaluator = BiasEvaluator()
        
        # Valid request
        request_data = {
            "counterfactuals": [
                {"image": "path1.jpg", "text": "text1", "attributes": {"gender": "female"}},
                {"image": "path2.jpg", "text": "text2", "attributes": {"gender": "male"}}
            ],
            "metrics": ["demographic_parity", "equalized_odds"],
            "protected_attributes": ["gender", "age"]
        }
        
        request = BiasEvaluationRequest(**request_data)
        assert len(request.counterfactuals) == 2
        assert "demographic_parity" in request.metrics
        
        # Mock response validation
        mock_response = {
            "bias_scores": {"gender": 0.15, "age": 0.08},
            "fairness_metrics": {"demographic_parity": 0.92, "equalized_odds": 0.88},
            "evaluation_time": 1.2,
            "recommendations": ["Consider balancing gender representation"]
        }
        
        response = BiasEvaluationResponse(**mock_response)
        assert response.bias_scores["gender"] == 0.15
        assert len(response.recommendations) == 1
    
    def test_contract_backward_compatibility(self):
        """Test that API contracts maintain backward compatibility."""
        # Test with v1.0 format
        v1_request = {
            "image_path": "test.jpg",
            "text": "A doctor",
            "attributes": ["gender"],
            "num_samples": 5
            # method field was added in v1.1, should default
        }
        
        request = CounterfactualRequest(**v1_request)
        assert request.method == "modicf"  # Default value maintained
        
        # Test that old response format still validates
        v1_response = {
            "counterfactuals": [{"image": "path.jpg", "text": "text"}],
            "metadata": {},
            "generation_time": 1.0,
            "method_used": "modicf"
        }
        
        response = CounterfactualResponse(**v1_response)
        assert response.method_used == "modicf"
    
    def test_contract_validation_errors(self):
        """Test that invalid requests are properly rejected."""
        # Missing required field
        with pytest.raises(ValidationError):
            CounterfactualRequest(
                text="A person",
                attributes=["gender"]
                # missing image_path
            )
        
        # Invalid field type
        with pytest.raises(ValidationError):
            CounterfactualRequest(
                image_path="test.jpg",
                text="A person", 
                attributes="gender",  # Should be List[str]
                num_samples=5
            )
        
        # Extra fields not allowed
        with pytest.raises(ValidationError):
            CounterfactualRequest(
                image_path="test.jpg",
                text="A person",
                attributes=["gender"],
                num_samples=5,
                extra_field="not_allowed"  # This should be rejected
            )
    
    def test_cross_version_compatibility(self):
        """Test API compatibility across different versions."""
        test_cases = [
            {
                "version": "1.0",
                "request": {
                    "image_path": "test.jpg",
                    "text": "A person",
                    "attributes": ["gender"]
                }
            },
            {
                "version": "1.1", 
                "request": {
                    "image_path": "test.jpg",
                    "text": "A person",
                    "attributes": ["gender"],
                    "method": "icg"
                }
            },
            {
                "version": "1.2",
                "request": {
                    "image_path": "test.jpg", 
                    "text": "A person",
                    "attributes": ["gender", "age"],
                    "num_samples": 10,
                    "method": "modicf"
                }
            }
        ]
        
        for case in test_cases:
            # All versions should validate successfully
            request = CounterfactualRequest(**case["request"])
            assert request.image_path == "test.jpg"
            print(f"Version {case['version']} compatibility: ✓")


class TestContractEvolution:
    """Test contract evolution and versioning strategies."""
    
    def test_additive_changes_compatibility(self):
        """Test that additive changes maintain compatibility."""
        # Old client using new API (should work)
        old_format_request = {
            "image_path": "test.jpg",
            "text": "A person",
            "attributes": ["gender"],
            "num_samples": 5
        }
        
        request = CounterfactualRequest(**old_format_request)
        assert request.method == "modicf"  # Uses default
    
    def test_deprecation_warnings(self):
        """Test deprecation warnings for old contract usage."""
        # This would be implemented with actual deprecation warnings
        # in the real API implementation
        pass
    
    def test_contract_documentation_generation(self):
        """Test that contracts can generate API documentation."""
        schema = CounterfactualRequest.schema()
        
        assert "properties" in schema
        assert "image_path" in schema["properties"]
        assert schema["properties"]["image_path"]["type"] == "string"
        
        # Test that schema includes validation constraints
        assert "required" in schema
        assert "image_path" in schema["required"]


def generate_contract_report():
    """Generate contract testing report for API documentation."""
    contracts = {
        "CounterfactualRequest": CounterfactualRequest.schema(),
        "CounterfactualResponse": CounterfactualResponse.schema(),
        "BiasEvaluationRequest": BiasEvaluationRequest.schema(),
        "BiasEvaluationResponse": BiasEvaluationResponse.schema()
    }
    
    report = {
        "api_version": "1.2",
        "contract_schemas": contracts,
        "compatibility_matrix": {
            "1.0": "full",
            "1.1": "full", 
            "1.2": "full"
        },
        "breaking_changes": [],
        "deprecation_warnings": []
    }
    
    with open("api_contract_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report


if __name__ == "__main__":
    # Generate contract documentation
    report = generate_contract_report()
    print("Contract testing report generated: api_contract_report.json")