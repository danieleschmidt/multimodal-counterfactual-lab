#!/usr/bin/env python3
"""
Repository maintenance automation script.
Handles dependency updates, cleanup tasks, and maintenance operations.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import re


class MaintenanceAutomator:
    """Automates repository maintenance tasks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.dry_run = False
        
    def run_command(self, command: str, capture_output: bool = True) -> Tuple[bool, str]:
        """Run shell command and return success status and output."""
        try:
            if self.dry_run:
                print(f"[DRY RUN] Would run: {command}")
                return True, ""
                
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output, 
                text=True,
                cwd=self.repo_path
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            print(f"Error running command '{command}': {e}")
            return False, ""
    
    def check_outdated_dependencies(self) -> List[Dict[str, str]]:
        """Check for outdated Python dependencies."""
        print("Checking for outdated dependencies...")
        
        success, output = self.run_command("pip list --outdated --format=json")
        if not success:
            print("Failed to check outdated dependencies")
            return []
        
        try:
            outdated = json.loads(output) if output else []
            print(f"Found {len(outdated)} outdated dependencies")
            return outdated
        except json.JSONDecodeError:
            print("Failed to parse outdated dependencies output")
            return []
    
    def update_dependencies(self, auto_update: bool = False) -> bool:
        """Update dependencies with safety checks."""
        outdated = self.check_outdated_dependencies()
        if not outdated:
            print("All dependencies are up to date")
            return True
        
        # Categorize updates by risk
        safe_updates = []
        risky_updates = []
        
        for dep in outdated:
            name = dep["name"]
            current = dep["version"]
            latest = dep["latest_version"]
            
            # Consider patch updates safe (same major.minor)
            current_parts = current.split(".")
            latest_parts = latest.split(".")
            
            if (len(current_parts) >= 2 and len(latest_parts) >= 2 and 
                current_parts[0] == latest_parts[0] and 
                current_parts[1] == latest_parts[1]):
                safe_updates.append(dep)
            else:
                risky_updates.append(dep)
        
        print(f"Safe updates (patch): {len(safe_updates)}")
        print(f"Risky updates (major/minor): {len(risky_updates)}")
        
        # Always apply safe updates
        for dep in safe_updates:
            print(f"Updating {dep['name']} from {dep['version']} to {dep['latest_version']}")
            success, _ = self.run_command(f"pip install {dep['name']}=={dep['latest_version']}")
            if not success:
                print(f"Failed to update {dep['name']}")
        
        # Apply risky updates only if auto_update is enabled
        if auto_update and risky_updates:
            print("Applying risky updates (auto-update enabled)...")
            for dep in risky_updates:
                print(f"Updating {dep['name']} from {dep['version']} to {dep['latest_version']}")
                success, _ = self.run_command(f"pip install {dep['name']}=={dep['latest_version']}")
                if not success:
                    print(f"Failed to update {dep['name']}")
        elif risky_updates:
            print("Risky updates found but auto-update disabled. Manual review required:")
            for dep in risky_updates:
                print(f"  - {dep['name']}: {dep['version']} -> {dep['latest_version']}")
        
        # Update requirements files
        self.update_requirements_files()
        
        return True
    
    def update_requirements_files(self) -> bool:
        """Update requirements files with current installed versions."""
        print("Updating requirements files...")
        
        # Update requirements.txt
        if (self.repo_path / "requirements.txt").exists():
            success, _ = self.run_command("pip freeze > requirements.txt")
            if success:
                print("Updated requirements.txt")
            else:
                print("Failed to update requirements.txt")
        
        # Update requirements-dev.txt if it exists
        if (self.repo_path / "requirements-dev.txt").exists():
            # This is more complex as it should only include dev dependencies
            print("requirements-dev.txt exists but automated update skipped (manual review needed)")
        
        return True
    
    def clean_build_artifacts(self) -> bool:
        """Clean up build artifacts and cache files."""
        print("Cleaning build artifacts...")
        
        patterns_to_clean = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/build",
            "**/dist",
            "**/*.egg-info",
            ".coverage",
            "coverage.xml",
            "coverage.json",
            ".tox",
            ".nox"
        ]
        
        for pattern in patterns_to_clean:
            files = list(self.repo_path.glob(pattern))
            for file_path in files:
                try:
                    if file_path.is_file():
                        if not self.dry_run:
                            file_path.unlink()
                        print(f"Removed file: {file_path}")
                    elif file_path.is_dir():
                        if not self.dry_run:
                            import shutil
                            shutil.rmtree(file_path)
                        print(f"Removed directory: {file_path}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
        
        return True
    
    def check_security_issues(self) -> List[Dict]:
        """Check for security issues in dependencies."""
        print("Checking for security issues...")
        
        # Run safety check
        success, output = self.run_command("safety check --json")
        issues = []
        
        if success and output:
            try:
                safety_data = json.loads(output)
                issues.extend(safety_data)
            except json.JSONDecodeError:
                pass
        
        # Run bandit security check
        success, output = self.run_command("bandit -r src -f json")
        if success and output:
            try:
                bandit_data = json.loads(output)
                for result in bandit_data.get("results", []):
                    issues.append({
                        "type": "code_security",
                        "filename": result.get("filename"),
                        "line": result.get("line_number"),
                        "issue": result.get("issue_text"),
                        "severity": result.get("issue_severity")
                    })
            except json.JSONDecodeError:
                pass
        
        if issues:
            print(f"Found {len(issues)} security issues")
            for issue in issues[:5]:  # Show first 5
                print(f"  - {issue.get('issue', 'Security issue found')}")
        else:
            print("No security issues found")
        
        return issues
    
    def optimize_imports(self) -> bool:
        """Optimize Python imports using isort."""
        print("Optimizing imports...")
        
        success, _ = self.run_command("isort src tests --diff --check-only")
        if not success:
            print("Import optimization needed, running isort...")
            success, _ = self.run_command("isort src tests")
            if success:
                print("Imports optimized")
            else:
                print("Failed to optimize imports")
        else:
            print("Imports are already optimized")
        
        return success
    
    def format_code(self) -> bool:
        """Format code using black."""
        print("Formatting code...")
        
        success, _ = self.run_command("black --check src tests")
        if not success:
            print("Code formatting needed, running black...")
            success, _ = self.run_command("black src tests")
            if success:
                print("Code formatted")
            else:
                print("Failed to format code")
        else:
            print("Code is already formatted")
        
        return success
    
    def lint_code(self) -> bool:
        """Lint code using ruff."""
        print("Linting code...")
        
        success, output = self.run_command("ruff check src tests")
        if not success:
            print("Linting issues found:")
            print(output)
            
            # Try to fix automatically
            print("Attempting automatic fixes...")
            success, _ = self.run_command("ruff check --fix src tests")
            if success:
                print("Linting issues fixed automatically")
            else:
                print("Some linting issues require manual attention")
        else:
            print("No linting issues found")
        
        return success
    
    def update_changelog(self, version: Optional[str] = None) -> bool:
        """Update changelog with recent commits."""
        changelog_path = self.repo_path / "CHANGELOG.md"
        if not changelog_path.exists():
            print("CHANGELOG.md not found, skipping update")
            return True
        
        print("Updating changelog...")
        
        # Get recent commits since last tag
        success, last_tag = self.run_command("git describe --tags --abbrev=0")
        if not success:
            last_tag = ""
        
        if last_tag:
            success, commits = self.run_command(f"git log {last_tag}..HEAD --oneline")
        else:
            success, commits = self.run_command("git log --oneline -n 20")
        
        if not success or not commits:
            print("No recent commits found for changelog")
            return True
        
        # Parse commits by type
        features = []
        fixes = []
        other = []
        
        for commit in commits.split('\n'):
            if commit.strip():
                if 'feat:' in commit.lower() or 'feature:' in commit.lower():
                    features.append(commit.strip())
                elif 'fix:' in commit.lower() or 'bug:' in commit.lower():
                    fixes.append(commit.strip())
                else:
                    other.append(commit.strip())
        
        # Generate changelog entry
        today = datetime.now().strftime("%Y-%m-%d")
        version_header = f"## [{version or 'Unreleased'}] - {today}"
        
        changelog_entry = [version_header, ""]
        
        if features:
            changelog_entry.append("### Added")
            for feature in features:
                changelog_entry.append(f"- {feature}")
            changelog_entry.append("")
        
        if fixes:
            changelog_entry.append("### Fixed")
            for fix in fixes:
                changelog_entry.append(f"- {fix}")
            changelog_entry.append("")
        
        if other:
            changelog_entry.append("### Changed")
            for change in other:
                changelog_entry.append(f"- {change}")
            changelog_entry.append("")
        
        # Insert into changelog
        if not self.dry_run:
            with open(changelog_path, 'r') as f:
                existing_content = f.read()
            
            # Find insertion point (after # Changelog header)
            lines = existing_content.split('\n')
            insert_index = 0
            for i, line in enumerate(lines):
                if line.startswith('# ') and 'changelog' in line.lower():
                    insert_index = i + 2  # After header and empty line
                    break
            
            # Insert new entry
            new_lines = lines[:insert_index] + changelog_entry + lines[insert_index:]
            
            with open(changelog_path, 'w') as f:
                f.write('\n'.join(new_lines))
        
        print(f"Changelog updated with {len(features + fixes + other)} entries")
        return True
    
    def run_maintenance(self, tasks: List[str], auto_update: bool = False) -> bool:
        """Run specified maintenance tasks."""
        print(f"Running maintenance tasks: {', '.join(tasks)}")
        
        success = True
        
        if "dependencies" in tasks:
            success &= self.update_dependencies(auto_update)
        
        if "security" in tasks:
            issues = self.check_security_issues()
            if issues:
                success = False
        
        if "cleanup" in tasks:
            success &= self.clean_build_artifacts()
        
        if "format" in tasks:
            success &= self.format_code()
            success &= self.optimize_imports()
        
        if "lint" in tasks:
            success &= self.lint_code()
        
        if "changelog" in tasks:
            success &= self.update_changelog()
        
        return success


def main():
    parser = argparse.ArgumentParser(description="Repository maintenance automation")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--tasks", 
                       choices=["dependencies", "security", "cleanup", "format", "lint", "changelog", "all"],
                       nargs="+", default=["all"], help="Maintenance tasks to run")
    parser.add_argument("--auto-update", action="store_true", 
                       help="Automatically apply risky dependency updates")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if "all" in args.tasks:
        args.tasks = ["dependencies", "security", "cleanup", "format", "lint", "changelog"]
    
    automator = MaintenanceAutomator(args.repo_path)
    automator.dry_run = args.dry_run
    
    try:
        success = automator.run_maintenance(args.tasks, args.auto_update)
        
        if success:
            print("✅ All maintenance tasks completed successfully")
            sys.exit(0)
        else:
            print("❌ Some maintenance tasks failed or found issues")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during maintenance: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()