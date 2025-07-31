# GitHub Actions Workflow Setup

Due to GitHub security restrictions, workflow files cannot be automatically created by GitHub Apps. This guide provides the workflow templates and setup instructions.

## 🚀 Quick Setup

### Step 1: Copy Workflow Files

Copy the following files from `docs/github-workflows-templates/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/github-workflows-templates/ci.yml .github/workflows/
cp docs/github-workflows-templates/release.yml .github/workflows/
cp docs/github-workflows-templates/security.yml .github/workflows/
cp docs/github-workflows-templates/codeql-config.yml .github/
```

### Step 2: Configure Repository Secrets

Add these secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

#### Required for Release Workflow
- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `GITHUB_TOKEN`: Automatically provided by GitHub

#### Optional for Enhanced Features
- `CODECOV_TOKEN`: Codecov integration token
- `SLACK_WEBHOOK`: Slack notifications webhook
- `DISCORD_WEBHOOK`: Discord notifications webhook

### Step 3: Enable Workflows

1. Go to repository Settings → Actions → General
2. Allow "Allow all actions and reusable workflows"
3. Enable "Allow GitHub Actions to create and approve pull requests"

## 📋 Workflow Overview

### 🔄 CI Workflow (`ci.yml`)
**Triggers**: Push to main/develop, Pull Requests
**Features**:
- Multi-Python testing (3.10, 3.11, 3.12)
- Code quality checks (ruff, black, mypy)
- Unit and integration tests with coverage
- Security scanning (Bandit, Safety)
- Docker image building
- Codecov integration

**Estimated Runtime**: 8-12 minutes

### 🚀 Release Workflow (`release.yml`)
**Triggers**: Git tags (v*)
**Features**:
- Full test suite execution
- Security vulnerability scanning
- Package building and validation
- PyPI publishing
- Docker image building and publishing to GHCR
- GitHub release creation with changelog

**Estimated Runtime**: 15-20 minutes

### 🛡️ Security Workflow (`security.yml`)
**Triggers**: Weekly schedule, Push to main, Pull Requests
**Features**:
- CodeQL static analysis
- Dependency vulnerability scanning
- Container security scanning with Trivy
- Secret detection with TruffleHog
- SBOM (Software Bill of Materials) generation

**Estimated Runtime**: 10-15 minutes

## 🔧 Customization Options

### Modify Python Versions
Edit the matrix in `ci.yml` and `release.yml`:
```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]  # Add or remove versions
```

### Adjust Security Scanning
Modify `security.yml` to customize scanning:
```yaml
# Disable specific scans by commenting out jobs
# codeql:           # Static analysis
# dependency-scan:  # Dependency vulnerabilities  
# container-scan:   # Container vulnerabilities
# secrets-scan:     # Secret detection
# sbom-generation:  # Software Bill of Materials
```

### Configure Release Automation
Customize release behavior in `release.yml`:
```yaml
# Change container registry
registry: ghcr.io  # or docker.io, quay.io, etc.

# Modify PyPI publishing
uses: pypa/gh-action-pypi-publish@release/v1
with:
  repository_url: https://upload.pypi.org/legacy/  # or test.pypi.org
```

## 📊 Branch Protection Rules

Set up branch protection for `main`:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Configure:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - Select status checks: `test`, `security`, `docker`
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators

## 🔍 Monitoring Workflow Health

### GitHub Actions Dashboard
Monitor workflow runs at: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`

### Workflow Badges
Add badges to README.md:
```markdown
[![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/CI/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml)
[![Security](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Security/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/security.yml)
[![Release](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Release/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/release.yml)
```

### Notifications Setup
Configure workflow notifications:
```yaml
# Add to any workflow for Slack notifications
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 🚨 Troubleshooting

### Common Issues

#### Workflow Permission Errors
```yaml
# Add to workflow if permission errors occur
permissions:
  contents: read
  security-events: write
  actions: read
```

#### Large File Upload Failures
```yaml
# Increase timeout for large builds
timeout-minutes: 30  # Default is 6 hours
```

#### Cache Issues
```yaml
# Clear cache if builds fail
- name: Clear Cache
  run: |
    gh extension install actions/gh-actions-cache
    gh actions-cache delete --confirm
```

### Workflow Debugging
```yaml
# Add debug step to any workflow
- name: Debug Information
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "SHA: ${{ github.sha }}"
    echo "Actor: ${{ github.actor }}"
    env
```

## 📈 Performance Optimization

### Caching Strategy
All workflows include intelligent caching:
- **pip cache**: Python dependencies
- **Docker layer cache**: Container builds
- **GitHub Actions cache**: Build artifacts

### Parallel Execution
Workflows are optimized for parallel execution:
- Matrix builds run simultaneously
- Independent jobs run in parallel
- Conditional job execution based on changes

### Resource Allocation
```yaml
# For resource-intensive jobs
runs-on: ubuntu-latest-4-cores  # Use more powerful runners
```

## 🔄 Maintenance

### Weekly Tasks
- Review security scan results
- Update dependency versions
- Monitor workflow performance metrics

### Monthly Tasks
- Update workflow action versions
- Review and optimize build times
- Audit workflow permissions and secrets

### Quarterly Tasks
- Security audit of workflow configurations
- Performance optimization review
- Update Python version matrix as needed

For advanced workflow configurations and enterprise features, see the [GitHub Actions documentation](https://docs.github.com/en/actions).