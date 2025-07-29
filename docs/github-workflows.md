# GitHub Workflows Documentation

This document contains templates for GitHub Actions workflows that should be manually created by repository maintainers.

## Dependency Update Workflow

Create this file as `.github/workflows/dependency-update.yml`:

```yaml
name: Automated Dependency Updates

on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pip-tools
          pip install -e ".[dev]"
      
      - name: Update dependencies
        run: |
          pip-compile --upgrade pyproject.toml
          pip-compile --upgrade --extra dev pyproject.toml
      
      - name: Run tests
        run: |
          make test-fast
          make security-scan
      
      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'deps: update dependencies'
          title: 'Automated dependency updates'
          body: |
            ## Automated Dependency Updates
            
            This PR contains automated dependency updates.
            
            ### Changes Made
            - Updated Python dependencies to latest compatible versions
            - Ran security scans to ensure no vulnerabilities
            - Verified tests pass with updated dependencies
            
            ### Review Checklist
            - [ ] Review dependency changes for breaking changes
            - [ ] Verify all tests pass
            - [ ] Check for any new security vulnerabilities
            - [ ] Ensure performance is not degraded
          branch: automated/dependency-updates
          labels: |
            dependencies
            automated
```

## CI/CD Workflow

Create this file as `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run linting
      run: make lint
    
    - name: Run type checking
      run: make type-check
    
    - name: Run tests
      run: make test-all
    
    - name: Run security scan
      run: make security-scan

  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: make docker-build
    
    - name: Test Docker image
      run: |
        docker run --rm counterfactual-lab:latest python -c "import counterfactual_lab; print('Import successful')"
```

## Security Scanning Workflow

Create this file as `.github/workflows/security.yml`:

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install -e ".[dev]"
    
    - name: Run comprehensive security scan
      run: ./scripts/security_scan.sh
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: security_reports/
        retention-days: 30
```

## Setup Instructions

1. **Create the workflow files** by copying the YAML content above into the respective files in `.github/workflows/`

2. **Configure repository secrets** if needed for deployment or external services

3. **Adjust schedules and triggers** based on your team's needs

4. **Test workflows** using manual triggers first before relying on scheduled runs

5. **Monitor workflow runs** and adjust configurations as needed

## Required Permissions

Ensure your repository has the following permissions configured:
- Actions: Read and write permissions
- Contents: Write permission (for automated PRs)
- Pull requests: Write permission (for automated PRs)
- Issues: Write permission (if using issue automation)

## Notes

- These workflows are designed to work with the project structure and tooling
- Modify the Python versions in the matrix based on your support requirements  
- Add additional steps for deployment if needed
- Consider adding notification steps for critical failures