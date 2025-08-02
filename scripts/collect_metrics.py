#!/usr/bin/env python3
"""
Automated metrics collection script for project health monitoring.
This script collects various metrics about code quality, security, performance,
and project collaboration metrics.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import sys


class MetricsCollector:
    """Collects and analyzes project metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        
    def load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to file."""
        metrics["project"]["last_updated"] = datetime.now().isoformat()
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def run_command(self, command: str, capture_output: bool = True) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output, 
                text=True,
                cwd=self.repo_path
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception as e:
            print(f"Error running command '{command}': {e}")
            return None
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage (if pytest-cov is available)
        coverage_output = self.run_command("pytest --cov=src --cov-report=json")
        if coverage_output and Path("coverage.json").exists():
            try:
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                    metrics["test_coverage"] = {
                        "current": round(coverage_data.get("totals", {}).get("percent_covered", 0), 2),
                        "last_measured": datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"Error reading coverage data: {e}")
        
        # Lines of code
        loc_output = self.run_command("find src -name '*.py' -exec wc -l {} + | tail -1")
        if loc_output:
            try:
                total_lines = int(loc_output.split()[0])
                metrics["lines_of_code"] = {
                    "total": total_lines,
                    "last_counted": datetime.now().isoformat()
                }
            except (ValueError, IndexError):
                pass
        
        # Cyclomatic complexity (if radon is available)
        complexity_output = self.run_command("radon cc src --average --json")
        if complexity_output:
            try:
                complexity_data = json.loads(complexity_output)
                total_complexity = 0
                file_count = 0
                for file_data in complexity_data.values():
                    if isinstance(file_data, list):
                        for func in file_data:
                            total_complexity += func.get("complexity", 0)
                            file_count += 1
                
                if file_count > 0:
                    metrics["cyclomatic_complexity"] = {
                        "current": round(total_complexity / file_count, 2),
                        "last_measured": datetime.now().isoformat()
                    }
            except (json.JSONDecodeError, KeyError):
                pass
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        metrics = {}
        
        # Safety check for Python dependencies
        safety_output = self.run_command("safety check --json")
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                vulnerabilities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                
                for vuln in safety_data:
                    severity = vuln.get("severity", "low").lower()
                    if severity in vulnerabilities:
                        vulnerabilities[severity] += 1
                
                metrics["dependency_vulnerabilities"] = {
                    **vulnerabilities,
                    "last_scan": datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                pass
        
        # Bandit security scan
        bandit_output = self.run_command("bandit -r src -f json")
        if bandit_output:
            try:
                bandit_data = json.loads(bandit_output)
                vulnerabilities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                
                for result in bandit_data.get("results", []):
                    severity = result.get("issue_severity", "low").lower()
                    if severity in vulnerabilities:
                        vulnerabilities[severity] += 1
                
                metrics["vulnerabilities"] = {
                    **vulnerabilities,
                    "last_scan": datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                pass
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}
        
        # Test execution time
        start_time = time.time()
        test_result = self.run_command("pytest --tb=no -q")
        if test_result is not None:
            test_time = time.time() - start_time
            metrics["test_execution_time"] = {
                "current": round(test_time, 2),
                "last_measured": datetime.now().isoformat()
            }
        
        # Docker build time (if Dockerfile exists)
        if (self.repo_path / "Dockerfile").exists():
            start_time = time.time()
            build_result = self.run_command("docker build -t metrics-test .")
            if build_result is not None:
                build_time = time.time() - start_time
                metrics["docker_build_time"] = {
                    "current": round(build_time, 2),
                    "last_measured": datetime.now().isoformat()
                }
                # Clean up test image
                self.run_command("docker rmi metrics-test 2>/dev/null")
        
        return metrics
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        metrics = {}
        
        # Contributors
        contributors_output = self.run_command("git shortlog -sn --all")
        if contributors_output:
            contributor_count = len(contributors_output.strip().split('\n'))
            metrics["contributors"] = {
                "total": contributor_count,
                "last_counted": datetime.now().isoformat()
            }
        
        # Recent activity
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent_commits = self.run_command(f"git rev-list --count --since='{one_month_ago}' HEAD")
        if recent_commits:
            try:
                metrics["recent_activity"] = {
                    "commits_last_30_days": int(recent_commits),
                    "last_calculated": datetime.now().isoformat()
                }
            except ValueError:
                pass
        
        # Code churn (additions and deletions)
        churn_output = self.run_command(f"git log --since='{one_month_ago}' --numstat --pretty=format:''")
        if churn_output:
            lines = churn_output.strip().split('\n')
            additions = deletions = 0
            for line in lines:
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        additions += int(parts[0])
                        deletions += int(parts[1])
            
            metrics["code_churn"] = {
                "additions_last_30_days": additions,
                "deletions_last_30_days": deletions,
                "last_calculated": datetime.now().isoformat()
            }
        
        return metrics
    
    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation metrics."""
        metrics = {}
        
        # Count documentation files
        doc_files = list(self.repo_path.glob("**/*.md"))
        doc_files.extend(list(self.repo_path.glob("docs/**/*")))
        
        metrics["documentation_files"] = {
            "count": len(doc_files),
            "last_counted": datetime.now().isoformat()
        }
        
        # Check for key documentation files
        key_docs = ["README.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"]
        present_docs = sum(1 for doc in key_docs if (self.repo_path / doc).exists())
        
        metrics["key_documentation"] = {
            "present": present_docs,
            "total": len(key_docs),
            "percentage": round((present_docs / len(key_docs)) * 100, 2),
            "last_checked": datetime.now().isoformat()
        }
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics and update the metrics file."""
        print("Collecting project metrics...")
        
        current_metrics = self.load_current_metrics()
        
        # Ensure structure exists
        if "metrics" not in current_metrics:
            current_metrics["metrics"] = {}
        
        # Collect different types of metrics
        print("- Code quality metrics...")
        code_quality = self.collect_code_quality_metrics()
        if code_quality:
            current_metrics["metrics"]["code_quality"] = {
                **current_metrics["metrics"].get("code_quality", {}),
                **code_quality
            }
        
        print("- Security metrics...")
        security = self.collect_security_metrics()
        if security:
            current_metrics["metrics"]["security"] = {
                **current_metrics["metrics"].get("security", {}),
                **security
            }
        
        print("- Performance metrics...")
        performance = self.collect_performance_metrics()
        if performance:
            current_metrics["metrics"]["performance"] = {
                **current_metrics["metrics"].get("performance", {}),
                **performance
            }
        
        print("- Git repository metrics...")
        git_metrics = self.collect_git_metrics()
        if git_metrics:
            current_metrics["metrics"]["collaboration"] = {
                **current_metrics["metrics"].get("collaboration", {}),
                **git_metrics
            }
        
        print("- Documentation metrics...")
        doc_metrics = self.collect_documentation_metrics()
        if doc_metrics:
            current_metrics["metrics"]["documentation"] = {
                **current_metrics["metrics"].get("documentation", {}),
                **doc_metrics
            }
        
        return current_metrics
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable metrics report."""
        report = []
        report.append("# Project Metrics Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if "project" in metrics:
            project = metrics["project"]
            report.append(f"**Project:** {project.get('name', 'Unknown')}")
            report.append(f"**Version:** {project.get('version', 'Unknown')}")
            report.append("")
        
        metric_sections = metrics.get("metrics", {})
        
        # Code Quality
        if "code_quality" in metric_sections:
            report.append("## Code Quality")
            cq = metric_sections["code_quality"]
            if "test_coverage" in cq:
                coverage = cq["test_coverage"]["current"]
                report.append(f"- Test Coverage: {coverage}%")
            if "lines_of_code" in cq:
                loc = cq["lines_of_code"]["total"]
                report.append(f"- Lines of Code: {loc:,}")
            if "cyclomatic_complexity" in cq:
                complexity = cq["cyclomatic_complexity"]["current"]
                report.append(f"- Average Cyclomatic Complexity: {complexity}")
            report.append("")
        
        # Security
        if "security" in metric_sections:
            report.append("## Security")
            sec = metric_sections["security"]
            if "vulnerabilities" in sec:
                vuln = sec["vulnerabilities"]
                total_vulns = sum(v for k, v in vuln.items() if k != "last_scan")
                report.append(f"- Total Vulnerabilities: {total_vulns}")
                if total_vulns > 0:
                    report.append(f"  - Critical: {vuln.get('critical', 0)}")
                    report.append(f"  - High: {vuln.get('high', 0)}")
                    report.append(f"  - Medium: {vuln.get('medium', 0)}")
                    report.append(f"  - Low: {vuln.get('low', 0)}")
            report.append("")
        
        # Performance
        if "performance" in metric_sections:
            report.append("## Performance")
            perf = metric_sections["performance"]
            if "test_execution_time" in perf:
                test_time = perf["test_execution_time"]["current"]
                report.append(f"- Test Execution Time: {test_time}s")
            if "docker_build_time" in perf:
                build_time = perf["docker_build_time"]["current"]
                report.append(f"- Docker Build Time: {build_time}s")
            report.append("")
        
        # Collaboration
        if "collaboration" in metric_sections:
            report.append("## Collaboration")
            collab = metric_sections["collaboration"]
            if "contributors" in collab:
                contributors = collab["contributors"]["total"]
                report.append(f"- Total Contributors: {contributors}")
            if "recent_activity" in collab:
                commits = collab["recent_activity"]["commits_last_30_days"]
                report.append(f"- Commits (last 30 days): {commits}")
            report.append("")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--output", choices=["json", "report", "both"], default="both", 
                       help="Output format")
    parser.add_argument("--save", action="store_true", help="Save metrics to file")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.repo_path)
    
    try:
        metrics = collector.collect_all_metrics()
        
        if args.save:
            collector.save_metrics(metrics)
            print("Metrics saved to .github/project-metrics.json")
        
        if args.output in ["json", "both"]:
            print(json.dumps(metrics, indent=2))
        
        if args.output in ["report", "both"]:
            report = collector.generate_report(metrics)
            print(report)
            
    except Exception as e:
        print(f"Error collecting metrics: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()