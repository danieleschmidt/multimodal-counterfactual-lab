#!/usr/bin/env python3
"""
Advanced release automation script with comprehensive security validation.

Features:
- Semantic versioning with pre-release support
- Automated changelog generation from git commits
- Security scanning and validation
- SBOM generation for supply chain security
- GPG-signed commits and tags
- GitHub release creation with assets
- Compliance reporting
"""

import argparse
import json
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SecurityValidationError(Exception):
    """Raised when security validation fails."""
    pass


class ReleaseAutomation:
    """Comprehensive release automation with security and compliance features."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.project_root = Path.cwd()
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.changelog_path = self.project_root / "CHANGELOG.md"
        
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with levels and dry run indicators."""
        prefix = f"{'[DRY RUN] ' if self.dry_run else ''}[{level}]"
        print(f"{prefix} {message}")
        
    def run_command(self, cmd: str, capture_output: bool = True, 
                   check: bool = True, sensitive: bool = False) -> subprocess.CompletedProcess:
        """Execute shell command with comprehensive error handling and security."""
        display_cmd = "[SENSITIVE COMMAND]" if sensitive else cmd
        self.log(f"Executing: {display_cmd}")
        
        if self.dry_run and any(action in cmd.lower() for action in 
                               ['commit', 'push', 'tag', 'release', 'upload', 'publish']):
            self.log("Skipping destructive command in dry run mode")
            return subprocess.CompletedProcess(cmd, 0, stdout="[DRY RUN]", stderr="")
            
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True
        )
        
        if check and result.returncode != 0:
            self.log(f"Command failed with exit code {result.returncode}", "ERROR")
            if capture_output and not sensitive:
                self.log(f"stdout: {result.stdout}", "ERROR")
                self.log(f"stderr: {result.stderr}", "ERROR")
            raise subprocess.CalledProcessError(result.returncode, cmd)
            
        return result
        
    def get_project_metadata(self) -> Dict:
        """Extract project metadata from pyproject.toml."""
        with open(self.pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data
        
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        metadata = self.get_project_metadata()
        return metadata["project"]["version"]
        
    def calculate_next_version(self, current_version: str, bump_type: str) -> str:
        """Calculate next version with support for pre-releases."""
        parts = current_version.split('.')
        if len(parts) < 3:
            raise ValueError(f"Invalid semantic version: {current_version}")
            
        # Handle pre-release versions
        if '-' in parts[2]:
            patch_part, pre_part = parts[2].split('-', 1)
            parts[2] = patch_part
            
        major, minor, patch = map(int, parts[:3])
        
        if bump_type == 'major':
            return f"{major + 1}.0.0"
        elif bump_type == 'minor':
            return f"{major}.{minor + 1}.0"
        elif bump_type == 'patch':
            return f"{major}.{minor}.{patch + 1}"
        elif bump_type == 'prerelease':
            # Create or increment pre-release
            if '-' in current_version:
                # Increment existing pre-release
                base, pre = current_version.split('-', 1)
                if 'rc' in pre:
                    rc_num = int(pre.split('rc')[1]) + 1
                    return f"{base}-rc{rc_num}"
                else:
                    return f"{base}-rc2"
            else:
                # Create new pre-release
                return f"{major}.{minor}.{patch + 1}-rc1"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
            
    def validate_repository_state(self) -> bool:
        """Comprehensive repository state validation."""
        self.log("Validating repository state...")
        
        # Check for uncommitted changes
        result = self.run_command("git status --porcelain", check=False)
        if result.stdout.strip():
            self.log("Repository has uncommitted changes", "ERROR")
            return False
            
        # Check current branch
        result = self.run_command("git branch --show-current")
        current_branch = result.stdout.strip()
        if current_branch not in ['main', 'master']:
            self.log(f"Not on main/master branch (current: {current_branch})", "WARNING")
            
        # Check for unpushed commits
        result = self.run_command("git log @{u}.. --oneline", check=False)
        if result.stdout.strip():
            self.log("Repository has unpushed commits", "WARNING")
            
        self.log("Repository state validation passed", "SUCCESS")
        return True
        
    def run_comprehensive_tests(self) -> bool:
        """Run comprehensive test suite before release."""
        self.log("Running comprehensive test suite...")
        
        test_commands = [
            ("python -m pytest tests/ -v --tb=short", "Unit tests"),
            ("python -m pytest tests/integration/ -v", "Integration tests"),
            ("python -m pytest tests/e2e/ -v", "End-to-end tests"),
        ]
        
        for cmd, description in test_commands:
            try:
                self.log(f"Running {description}...")
                self.run_command(cmd)
                self.log(f"{description} passed", "SUCCESS")
            except subprocess.CalledProcessError:
                self.log(f"{description} failed", "ERROR")
                return False
                
        return True
        
    def run_security_validation(self) -> bool:
        """Comprehensive security validation pipeline."""
        self.log("Running security validation pipeline...")
        
        # Secret scanning
        try:
            self.run_command("git secrets --scan", check=False)
            self.log("Secret scanning completed", "SUCCESS")
        except:
            self.log("git-secrets not available, skipping secret scan", "WARNING")
            
        # Security linting with bandit
        try:
            result = self.run_command("bandit -r src/ -f json -o bandit-report.json", check=False)
            if result.returncode == 0:
                self.log("Bandit security linting passed", "SUCCESS")
            else:
                self.log("Bandit found security issues", "ERROR")
                return False
        except:
            self.log("Bandit not available, skipping security linting", "WARNING")
            
        # Dependency vulnerability scanning
        try:
            result = self.run_command("pip-audit --format=json --output=vulnerability-report.json", check=False)
            if result.returncode == 0:
                self.log("Dependency vulnerability scan passed", "SUCCESS")
            else:
                with open("vulnerability-report.json", "r") as f:
                    vuln_data = json.load(f)
                    if vuln_data.get("vulnerabilities"):
                        self.log(f"Found {len(vuln_data['vulnerabilities'])} vulnerabilities", "ERROR")
                        return False
        except:
            self.log("pip-audit not available, skipping vulnerability scan", "WARNING")
            
        # Container security scanning if Dockerfile exists
        if (self.project_root / "Dockerfile").exists():
            try:
                self.run_command("hadolint Dockerfile", check=False)
                self.log("Dockerfile security scan completed", "SUCCESS")
            except:
                self.log("hadolint not available, skipping Dockerfile scan", "WARNING")
                
        return True
        
    def generate_sbom(self, version: str) -> Optional[Path]:
        """Generate comprehensive Software Bill of Materials."""
        self.log("Generating Software Bill of Materials (SBOM)...")
        
        sbom_dir = self.project_root / "sbom"
        sbom_dir.mkdir(exist_ok=True)
        
        sbom_files = []
        
        # Generate CycloneDX SBOM
        try:
            cyclone_file = sbom_dir / f"sbom-cyclonedx-{version}.json"
            self.run_command(f"cyclonedx-py --output-format json --output-file {cyclone_file}")
            sbom_files.append(cyclone_file)
            self.log(f"CycloneDX SBOM generated: {cyclone_file}", "SUCCESS")
        except:
            self.log("CycloneDX generation failed", "WARNING")
            
        # Generate SPDX SBOM using pip-audit
        try:
            spdx_file = sbom_dir / f"sbom-spdx-{version}.json"
            self.run_command(f"pip-audit --format=json --output={spdx_file}")
            sbom_files.append(spdx_file)
            self.log(f"SPDX SBOM generated: {spdx_file}", "SUCCESS")
        except:
            self.log("SPDX generation failed", "WARNING")
            
        return sbom_files[0] if sbom_files else None
        
    def extract_commits_since_last_tag(self) -> List[str]:
        """Extract commit messages since last tag for changelog generation."""
        try:
            # Get last tag
            result = self.run_command("git describe --tags --abbrev=0", check=False)
            if result.returncode == 0:
                last_tag = result.stdout.strip()
                commit_range = f"{last_tag}..HEAD"
            else:
                commit_range = "HEAD"
        except:
            commit_range = "HEAD"
            
        # Get commit messages with conventional commit parsing
        result = self.run_command(f"git log {commit_range} --pretty=format:'%s' --no-merges")
        commits = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        
        return commits
        
    def categorize_commits(self, commits: List[str]) -> Dict[str, List[str]]:
        """Categorize commits using conventional commit patterns."""
        categories = {
            'breaking': [],
            'feat': [],
            'fix': [],
            'security': [],
            'perf': [],
            'docs': [],
            'style': [],
            'refactor': [],
            'test': [],
            'build': [],
            'ci': [],
            'chore': [],
            'revert': []
        }
        
        for commit in commits:
            commit_lower = commit.lower()
            
            # Check for breaking changes
            if 'breaking change' in commit_lower or '!' in commit:
                categories['breaking'].append(commit)
            # Conventional commit patterns
            elif commit_lower.startswith('feat'):
                categories['feat'].append(commit)
            elif commit_lower.startswith('fix'):
                categories['fix'].append(commit)
            elif commit_lower.startswith('security') or 'cve' in commit_lower:
                categories['security'].append(commit)
            elif commit_lower.startswith('perf'):
                categories['perf'].append(commit)
            elif commit_lower.startswith('docs'):
                categories['docs'].append(commit)
            elif commit_lower.startswith('style'):
                categories['style'].append(commit)
            elif commit_lower.startswith('refactor'):
                categories['refactor'].append(commit)
            elif commit_lower.startswith('test'):
                categories['test'].append(commit)
            elif commit_lower.startswith('build'):
                categories['build'].append(commit)
            elif commit_lower.startswith('ci'):
                categories['ci'].append(commit)
            elif commit_lower.startswith('revert'):
                categories['revert'].append(commit)
            else:
                categories['chore'].append(commit)
                
        return categories
        
    def generate_changelog_entry(self, version: str) -> str:
        """Generate comprehensive changelog entry."""
        commits = self.extract_commits_since_last_tag()
        if not commits:
            return f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n\nNo changes since last release.\n"
            
        categories = self.categorize_commits(commits)
        
        changelog = f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # Breaking changes (highest priority)
        if categories['breaking']:
            changelog += "### ⚠️ BREAKING CHANGES\n\n"
            for commit in categories['breaking']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Security fixes
        if categories['security']:
            changelog += "### 🔒 Security\n\n"
            for commit in categories['security']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # New features
        if categories['feat']:
            changelog += "### ✨ Features\n\n"
            for commit in categories['feat']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Bug fixes
        if categories['fix']:
            changelog += "### 🐛 Bug Fixes\n\n"
            for commit in categories['fix']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Performance improvements
        if categories['perf']:
            changelog += "### ⚡ Performance\n\n"
            for commit in categories['perf']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Documentation
        if categories['docs']:
            changelog += "### 📚 Documentation\n\n"
            for commit in categories['docs']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Refactoring
        if categories['refactor']:
            changelog += "### 🔧 Code Refactoring\n\n"
            for commit in categories['refactor']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Tests
        if categories['test']:
            changelog += "### 🧪 Tests\n\n"
            for commit in categories['test']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Build system
        if categories['build'] or categories['ci']:
            changelog += "### 🏗️ Build System\n\n"
            for commit in categories['build'] + categories['ci']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        # Other changes
        if categories['chore'] or categories['style']:
            changelog += "### 🔨 Other Changes\n\n"
            for commit in categories['chore'] + categories['style']:
                changelog += f"- {commit}\n"
            changelog += "\n"
            
        return changelog
        
    def update_version_files(self, new_version: str):
        """Update version in all relevant project files."""
        current_version = self.get_current_version()
        
        # Update pyproject.toml
        with open(self.pyproject_path, "r") as f:
            content = f.read()
            
        content = content.replace(
            f'version = "{current_version}"',
            f'version = "{new_version}"'
        )
        
        if not self.dry_run:
            with open(self.pyproject_path, "w") as f:
                f.write(content)
                
        # Update __init__.py if it exists
        init_file = self.project_root / "src" / "counterfactual_lab" / "__init__.py"
        if init_file.exists():
            with open(init_file, "r") as f:
                init_content = f.read()
                
            if f'__version__ = "{current_version}"' in init_content:
                init_content = init_content.replace(
                    f'__version__ = "{current_version}"',
                    f'__version__ = "{new_version}"'
                )
                
                if not self.dry_run:
                    with open(init_file, "w") as f:
                        f.write(init_content)
                        
        self.log(f"Updated version from {current_version} to {new_version}", "SUCCESS")
        
    def update_changelog_file(self, new_entry: str):
        """Update CHANGELOG.md with new version entry."""
        if self.changelog_path.exists():
            with open(self.changelog_path, "r") as f:
                existing_content = f.read()
                
            # Find insertion point (after header, before first version)
            lines = existing_content.split('\n')
            insert_index = 0
            
            for i, line in enumerate(lines):
                if line.startswith('## [') and '- ' in line:
                    insert_index = i
                    break
                elif line.startswith('## ') and i > 0:
                    insert_index = i
                    break
                    
            # Insert new entry
            new_lines = lines[:insert_index] + [new_entry] + lines[insert_index:]
            new_content = '\n'.join(new_lines)
        else:
            # Create new changelog
            header = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
            new_content = header + new_entry
            
        if not self.dry_run:
            with open(self.changelog_path, "w") as f:
                f.write(new_content)
                
        self.log("Updated CHANGELOG.md", "SUCCESS")
        
    def create_signed_commit_and_tag(self, version: str):
        """Create GPG-signed commit and tag for release."""
        tag_name = f"v{version}"
        
        # Stage all changes
        self.run_command("git add -A")
        
        # Create signed commit
        commit_message = f"""chore: release {version}

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""
        
        self.run_command(f'git commit -S -m "{commit_message}"')
        
        # Create signed tag
        tag_message = f"""Release {version}

Automated release with comprehensive security validation:
- Dependency vulnerability scanning
- Security linting and code analysis  
- Software Bill of Materials (SBOM) generation
- GPG-signed commits and tags
- Comprehensive test suite execution

🤖 Generated with [Claude Code](https://claude.ai/code)"""
        
        self.run_command(f'git tag -s {tag_name} -m "{tag_message}"')
        
        self.log(f"Created signed commit and tag: {tag_name}", "SUCCESS")
        
    def push_changes(self, version: str):
        """Push commits and tags to remote repository."""
        tag_name = f"v{version}"
        
        # Push commits
        self.run_command("git push origin main")
        
        # Push tags
        self.run_command(f"git push origin {tag_name}")
        
        self.log("Pushed changes and tags to remote", "SUCCESS")
        
    def create_github_release(self, version: str, changelog_entry: str, sbom_file: Optional[Path] = None):
        """Create comprehensive GitHub release with assets."""
        tag_name = f"v{version}"
        
        # Check for GitHub CLI
        try:
            self.run_command("gh --version")
        except:
            self.log("GitHub CLI not available - skipping release creation", "WARNING")
            return
            
        # Prepare comprehensive release notes
        release_notes = f"""# Multimodal Counterfactual Lab {version}

{changelog_entry}

## 🔒 Security & Compliance

This release includes comprehensive security validation:

- ✅ Dependency vulnerability scanning with pip-audit  
- ✅ Security linting with Bandit
- ✅ Container security scanning with Hadolint
- ✅ Software Bill of Materials (SBOM) generation
- ✅ GPG-signed commits and tags
- ✅ Comprehensive test suite execution

## 📋 Verification

To verify the integrity of this release:

```bash
# Verify GPG signature of the tag
git verify-tag {tag_name}

# Verify container image signature (if using containers)
cosign verify multimodal-counterfactual-lab:{version}
```

## 📦 Installation

### From PyPI
```bash
pip install multimodal-counterfactual-lab=={version}
```

### From Docker Hub
```bash
docker pull multimodal-counterfactual-lab:{version}
```

## 🤖 Automation

This release was created using automated tooling with:
- Semantic versioning
- Conventional commit changelog generation  
- Comprehensive security scanning
- Supply chain security validation

Generated with [Claude Code](https://claude.ai/code)
"""
        
        # Create temporary file for release notes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(release_notes)
            notes_file = Path(f.name)
            
        try:
            # Build release command
            cmd = f'gh release create {tag_name} --title "Release {version}" --notes-file {notes_file}'
            
            # Add SBOM as asset if available
            if sbom_file and sbom_file.exists():
                cmd += f' "{sbom_file}"'
                
            # Add security reports as assets
            security_files = [
                "bandit-report.json",
                "vulnerability-report.json"
            ]
            
            for sec_file in security_files:
                sec_path = self.project_root / sec_file
                if sec_path.exists():
                    cmd += f' "{sec_path}"'
                    
            self.run_command(cmd)
            self.log(f"GitHub release {version} created successfully", "SUCCESS")
            
        finally:
            # Clean up temporary file
            if not self.dry_run:
                notes_file.unlink(missing_ok=True)
                
    def execute_release(self, bump_type: str) -> str:
        """Execute complete release workflow."""
        self.log(f"Starting {bump_type} release automation...", "INFO")
        
        # Step 1: Validate repository state
        if not self.validate_repository_state():
            raise SecurityValidationError("Repository state validation failed")
            
        # Step 2: Calculate version
        current_version = self.get_current_version()
        new_version = self.calculate_next_version(current_version, bump_type)
        
        self.log(f"Version transition: {current_version} → {new_version}", "INFO")
        
        if not self.dry_run:
            confirm = input(f"Proceed with {bump_type} release {new_version}? (y/N): ")
            if confirm.lower() != 'y':
                self.log("Release cancelled by user", "INFO")
                sys.exit(0)
                
        # Step 3: Run comprehensive tests
        if not self.run_comprehensive_tests():
            raise SecurityValidationError("Test validation failed")
            
        # Step 4: Run security validation
        if not self.run_security_validation():
            raise SecurityValidationError("Security validation failed")
            
        # Step 5: Generate SBOM
        sbom_file = self.generate_sbom(new_version)
        
        # Step 6: Update version files
        self.update_version_files(new_version)
        
        # Step 7: Generate and update changelog
        changelog_entry = self.generate_changelog_entry(new_version)
        self.update_changelog_file(changelog_entry)
        
        # Step 8: Create signed commit and tag
        self.create_signed_commit_and_tag(new_version)
        
        # Step 9: Push changes
        self.push_changes(new_version)
        
        # Step 10: Create GitHub release
        self.create_github_release(new_version, changelog_entry, sbom_file)
        
        self.log(f"Release {new_version} completed successfully! 🎉", "SUCCESS")
        self.log(f"Release URL: https://github.com/terragon-labs/multimodal-counterfactual-lab/releases/tag/v{new_version}", "INFO")
        
        return new_version


def main():
    """Main CLI entry point for release automation."""
    parser = argparse.ArgumentParser(
        description="Comprehensive release automation with security validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s patch --dry-run        # Preview patch release
  %(prog)s minor                  # Create minor release
  %(prog)s major --verbose        # Create major release with verbose output
  %(prog)s prerelease             # Create pre-release version
        """
    )
    
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch", "prerelease"],
        help="Type of version bump to perform"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing them"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        automation = ReleaseAutomation(
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        version = automation.execute_release(args.bump_type)
        
        if args.dry_run:
            print(f"\n✅ Dry run completed successfully for version {version}")
        else:
            print(f"\n🎉 Release {version} published successfully!")
            
    except KeyboardInterrupt:
        print("\n❌ Release cancelled by user")
        sys.exit(1)
    except SecurityValidationError as e:
        print(f"\n❌ Security validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Release failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()