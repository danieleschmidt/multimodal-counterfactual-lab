#!/usr/bin/env python3
"""Release preparation script for Multimodal Counterfactual Lab."""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import toml


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    with open(pyproject_path) as f:
        data = toml.load(f)
    
    return data["project"]["version"]


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version according to semantic versioning."""
    major, minor, patch = map(int, current_version.split("."))
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_version_in_files(new_version: str) -> None:
    """Update version in relevant files."""
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path) as f:
        content = f.read()
    
    content = re.sub(
        r'version = "[^"]+"',
        f'version = "{new_version}"',
        content
    )
    
    with open(pyproject_path, "w") as f:
        f.write(content)
    
    # Update __init__.py
    init_path = Path("src/counterfactual_lab/__init__.py")
    if init_path.exists():
        with open(init_path) as f:
            content = f.read()
        
        content = re.sub(
            r'__version__ = "[^"]+"',
            f'__version__ = "{new_version}"',
            content
        )
        
        with open(init_path, "w") as f:
            f.write(content)


def update_changelog(version: str) -> None:
    """Update CHANGELOG.md with new version."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("⚠️  CHANGELOG.md not found, skipping changelog update")
        return
    
    with open(changelog_path) as f:
        content = f.read()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Replace [Unreleased] with version and date
    content = re.sub(
        r"## \[Unreleased\]",
        f"## [Unreleased]\n\n## [{version}] - {today}",
        content
    )
    
    with open(changelog_path, "w") as f:
        f.write(content)


def run_quality_checks() -> bool:
    """Run quality checks before release."""
    print("🔍 Running quality checks...")
    
    checks = [
        ("make lint", "Code formatting and linting"),
        ("make type-check", "Type checking"),
        ("make test", "Unit tests"),
        ("./scripts/security_scan.sh", "Security scanning")
    ]
    
    for command, description in checks:
        print(f"  Running {description}...")
        try:
            subprocess.run(command.split(), check=True, capture_output=True)
            print(f"  ✅ {description} passed")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {description} failed:")
            print(f"    {e.stderr.decode()}")
            return False
    
    return True


def create_git_tag(version: str, dry_run: bool = False) -> None:
    """Create git tag for the release."""
    tag_name = f"v{version}"
    commit_message = f"Release {version}"
    
    if dry_run:
        print(f"Would create git tag: {tag_name}")
        return
    
    # Commit version changes
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", commit_message], check=True)
    
    # Create annotated tag
    subprocess.run([
        "git", "tag", "-a", tag_name, "-m", f"Release version {version}"
    ], check=True)
    
    print(f"✅ Created git tag: {tag_name}")


def generate_release_notes(version: str) -> str:
    """Generate release notes from changelog."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        return f"Release {version}"
    
    with open(changelog_path) as f:
        content = f.read()
    
    # Extract section for this version
    pattern = rf"## \[{re.escape(version)}\][^#]*?(?=## \[|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(0).strip()
    else:
        return f"Release {version}"


def main():
    """Main release preparation function."""
    parser = argparse.ArgumentParser(description="Prepare a new release")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip quality checks (not recommended)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Preparing release...")
    
    # Get current version and calculate new version
    current_version = get_current_version()
    new_version = bump_version(current_version, args.bump_type)
    
    print(f"📦 Version: {current_version} → {new_version}")
    
    if args.dry_run:
        print("🔍 DRY RUN - No changes will be made")
    
    # Run quality checks
    if not args.skip_checks:
        if not run_quality_checks():
            print("❌ Quality checks failed. Fix issues before releasing.")
            sys.exit(1)
    else:
        print("⚠️  Skipping quality checks")
    
    # Update version in files
    if not args.dry_run:
        update_version_in_files(new_version)
        update_changelog(new_version)
        print("✅ Updated version in files")
    else:
        print("Would update version in pyproject.toml and __init__.py")
    
    # Create git tag
    create_git_tag(new_version, args.dry_run)
    
    # Generate release notes
    release_notes = generate_release_notes(new_version)
    
    print("\n📝 Release Notes:")
    print("=" * 50)
    print(release_notes)
    print("=" * 50)
    
    if not args.dry_run:
        print(f"\n✅ Release {new_version} prepared successfully!")
        print("\nNext steps:")
        print("1. Review the changes and release notes")
        print("2. Push the changes: git push && git push --tags")
        print("3. Create a GitHub release using the generated notes")
        print("4. Build and publish to PyPI: make build && make publish")
    else:
        print(f"\n✅ Dry run completed for release {new_version}")


if __name__ == "__main__":
    main()