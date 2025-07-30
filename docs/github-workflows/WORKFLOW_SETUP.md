# GitHub Workflows Setup Guide

## Overview

This directory contains the GitHub Actions workflow files that need to be manually added to `.github/workflows/` due to GitHub App permission restrictions for workflow modifications.

## Required Workflows

### 1. **ci.yml** - Continuous Integration Pipeline
**Location**: `.github/workflows/ci.yml`

**Features**:
- Multi-OS testing (Ubuntu, macOS, Windows)
- Python 3.10, 3.11, 3.12 compatibility matrix
- Comprehensive testing (unit, integration, performance)
- Code quality enforcement (linting, type checking, security)
- Coverage reporting with Codecov integration

**Setup Instructions**:
1. Copy `docs/github-workflows/ci.yml` to `.github/workflows/ci.yml`
2. Ensure CODECOV_TOKEN is configured in repository secrets
3. Verify workflow permissions are enabled in repository settings

### 2. **security.yml** - Security Scanning Pipeline
**Location**: `.github/workflows/security.yml`

**Features**:
- Dependency vulnerability scanning (Safety, pip-audit)
- Static code analysis (CodeQL)
- Container security scanning (Trivy)
- Secret detection (TruffleHog)
- Automated security reporting

**Setup Instructions**:
1. Copy `docs/github-workflows/security.yml` to `.github/workflows/security.yml`
2. Enable CodeQL analysis in repository security settings
3. Configure security alerts and notifications

### 3. **release.yml** - Automated Release Pipeline
**Location**: `.github/workflows/release.yml`

**Features**:
- Automated PyPI publishing on version tags
- Multi-architecture Docker image builds
- GitHub Releases creation with changelogs
- Version validation and artifact verification

**Setup Instructions**:
1. Copy `docs/github-workflows/release.yml` to `.github/workflows/release.yml`
2. Configure repository secrets:
   - `PYPI_API_TOKEN` - For PyPI publishing
   - `DOCKERHUB_USERNAME` - For Docker Hub
   - `DOCKERHUB_TOKEN` - For Docker Hub authentication
3. Enable trusted publishing on PyPI (recommended)

## Repository Secrets Configuration

### Required Secrets
```bash
# PyPI Publishing
PYPI_API_TOKEN=pypi-xxxxxxxxxxxxx

# Docker Hub
DOCKERHUB_USERNAME=your-dockerhub-username
DOCKERHUB_TOKEN=your-dockerhub-token

# Codecov (optional but recommended)
CODECOV_TOKEN=your-codecov-token
```

### Setting Up Secrets
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each required secret with appropriate values

## Workflow Permissions

### Required Permissions
Ensure the following permissions are enabled in repository settings:

**Actions Permissions**:
- ✅ Allow all actions and reusable workflows
- ✅ Allow actions created by GitHub
- ✅ Allow specified actions and reusable workflows

**Workflow Permissions**:
- ✅ Read and write permissions
- ✅ Allow GitHub Actions to create and approve pull requests

## Manual Setup Steps

### 1. Copy Workflow Files
```bash
# Navigate to your repository root
cd /path/to/your/repository

# Create workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy all workflow files
cp docs/github-workflows/ci.yml .github/workflows/
cp docs/github-workflows/security.yml .github/workflows/
cp docs/github-workflows/release.yml .github/workflows/
```

### 2. Configure Repository Settings
1. **Enable Actions**: Repository Settings → Actions → General
2. **Set Permissions**: Actions → General → Workflow permissions
3. **Enable Security**: Security → Code security and analysis
4. **Configure Branches**: Settings → Branches → Branch protection rules

### 3. Test Workflows
```bash
# Create a test commit to trigger CI
git add .
git commit -m "feat: enable GitHub Actions workflows"
git push

# Check workflow execution
# Go to repository → Actions tab to monitor workflow runs
```

## Workflow Customization

### Environment-Specific Configuration
Each workflow can be customized for your specific needs:

**CI Workflow Customization**:
- Modify Python versions in matrix strategy
- Add/remove testing environments
- Adjust coverage thresholds
- Configure additional quality gates

**Security Workflow Customization**:
- Adjust scan frequency (currently weekly)
- Configure severity thresholds
- Add custom security tools
- Set up notification channels

**Release Workflow Customization**:
- Modify release triggers (tags, branches)
- Configure artifact destinations
- Add custom build steps
- Set up deployment environments

## Troubleshooting

### Common Issues

**1. Workflow Permission Errors**
```
Error: Resource not accessible by integration
```
**Solution**: Enable workflow permissions in repository settings

**2. Secret Not Found Errors**
```
Error: Secret PYPI_API_TOKEN not found
```
**Solution**: Configure required secrets in repository settings

**3. CodeQL Analysis Failures**
```
Error: CodeQL analysis failed
```
**Solution**: Ensure CodeQL is enabled in repository security settings

### Verification Steps

**Verify CI Pipeline**:
1. Create a pull request
2. Check that all CI checks pass
3. Verify test coverage reports
4. Confirm security scans complete

**Verify Release Pipeline**:
1. Create a version tag (e.g., v1.0.0)
2. Check PyPI package publication
3. Verify Docker image builds
4. Confirm GitHub Release creation

## Support

For workflow setup assistance:
- **Primary**: daniel@terragon.ai
- **Documentation**: This setup guide and workflow comments
- **GitHub Actions Docs**: https://docs.github.com/en/actions

## Next Steps After Setup

1. **Monitor First Runs**: Check workflow execution and resolve any issues
2. **Configure Notifications**: Set up Slack/email alerts for failures
3. **Optimize Performance**: Adjust caching and parallelization
4. **Set Up Deployment**: Configure production deployment pipelines
5. **Enable Dependabot**: Ensure automated dependency updates work correctly

The workflows are designed to work seamlessly with the enhanced repository structure and will provide comprehensive automation for development, security, and release processes.