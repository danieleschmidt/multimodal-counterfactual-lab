"""Mutation testing configuration for enhanced test quality validation."""

import pytest
from mutmut import mutate_file, run_mutation_tests


class TestMutationTesting:
    """Enhanced testing with mutation testing for critical components."""
    
    def test_core_mutation_resistance(self):
        """Test that core counterfactual generation logic is mutation-resistant."""
        target_files = [
            "src/counterfactual_lab/core.py",
            "src/counterfactual_lab/methods/modicf.py",
            "src/counterfactual_lab/methods/icg.py"
        ]
        
        for file_path in target_files:
            # Run mutation testing on critical components
            result = run_mutation_tests(
                target=file_path,
                test_path="tests/unit/",
                mutation_threshold=0.8  # 80% mutation kill rate required
            )
            
            assert result.kill_rate >= 0.8, f"Mutation kill rate too low for {file_path}: {result.kill_rate}"
    
    def test_bias_evaluation_mutation_resistance(self):
        """Ensure bias evaluation logic is robust against mutations."""
        result = run_mutation_tests(
            target="src/counterfactual_lab/core.py:BiasEvaluator",
            test_path="tests/unit/test_core.py",
            mutation_threshold=0.85
        )
        
        assert result.kill_rate >= 0.85, "Bias evaluation mutation resistance insufficient"
    
    def test_fairness_metrics_mutation_coverage(self):
        """Test mutation coverage for fairness metric calculations.""" 
        # Target fairness-critical code paths
        fairness_modules = [
            "src/counterfactual_lab/metrics/",
            "src/counterfactual_lab/evaluation/"
        ]
        
        for module in fairness_modules:
            result = run_mutation_tests(
                target=module,
                test_path="tests/integration/",
                mutation_threshold=0.75
            )
            
            assert result.kill_rate >= 0.75, f"Fairness metrics mutation coverage low: {module}"


# Mutation test configuration
MUTATION_CONFIG = {
    "target_patterns": [
        "src/counterfactual_lab/core.py",
        "src/counterfactual_lab/methods/*.py",
        "src/counterfactual_lab/metrics/*.py"
    ],
    "excluded_patterns": [
        "*/test_*.py",
        "*/conftest.py",
        "*/__init__.py"
    ],
    "minimum_kill_rate": 0.80,
    "timeout": 300,  # 5 minutes per mutation
    "parallel_workers": 4
}


def run_comprehensive_mutation_tests():
    """Run mutation testing across all critical modules."""
    results = {}
    
    for pattern in MUTATION_CONFIG["target_patterns"]:
        result = run_mutation_tests(
            target=pattern,
            test_path="tests/",
            config=MUTATION_CONFIG
        )
        results[pattern] = result
    
    # Generate mutation testing report
    generate_mutation_report(results)
    
    return results


def generate_mutation_report(results):
    """Generate comprehensive mutation testing report."""
    report = {
        "summary": {
            "total_mutations": sum(r.total_mutations for r in results.values()),
            "killed_mutations": sum(r.killed_mutations for r in results.values()),
            "survived_mutations": sum(r.survived_mutations for r in results.values()),
            "overall_kill_rate": sum(r.killed_mutations for r in results.values()) / 
                               sum(r.total_mutations for r in results.values())
        },
        "by_module": {module: r.to_dict() for module, r in results.items()},
        "recommendations": generate_mutation_recommendations(results)
    }
    
    with open("mutation_test_report.json", "w") as f:
        import json
        json.dump(report, f, indent=2)


def generate_mutation_recommendations(results):
    """Generate recommendations based on mutation testing results."""
    recommendations = []
    
    for module, result in results.items():
        if result.kill_rate < MUTATION_CONFIG["minimum_kill_rate"]:
            recommendations.append({
                "module": module,
                "issue": "Low mutation kill rate",
                "current_rate": result.kill_rate,
                "target_rate": MUTATION_CONFIG["minimum_kill_rate"],
                "action": "Add more targeted unit tests for edge cases"
            })
    
    return recommendations


if __name__ == "__main__":
    # Run mutation tests when executed directly
    results = run_comprehensive_mutation_tests()
    print(f"Mutation testing complete. Overall kill rate: {results}")