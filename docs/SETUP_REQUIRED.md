# Manual Setup Required

Due to GitHub App permission limitations, some setup steps must be completed manually by repository maintainers.

## Required Manual Actions

### 1. GitHub Workflows
GitHub workflows need to be created manually from the templates provided in `docs/github-workflows-templates/`:

```bash
# Copy workflow templates to .github/workflows/
mkdir -p .github/workflows
cp docs/github-workflows-templates/*.yml .github/workflows/
```

**Templates available:**
- `ci.yml` - Continuous Integration (testing, linting, security)
- `security.yml` - Security scanning and dependency checks
- `release.yml` - Automated releases and changelog generation
- `codeql-config.yml` - GitHub CodeQL security analysis

### 2. Repository Settings

#### Branch Protection Rules
Navigate to **Settings → Branches → Add rule** and configure:

```yaml
Branch name pattern: main
Protect matching branches: ✓
Settings:
  - Require pull request reviews before merging: ✓
    - Required approving reviews: 1
    - Dismiss stale reviews: ✓
    - Require review from code owners: ✓
  - Require status checks before merging: ✓
    - Require branches to be up to date: ✓
    - Status checks: CI, Security Scan, Tests
  - Require conversation resolution before merging: ✓
  - Restrict pushes that create files larger than 100MB: ✓
```

#### Repository Topics
Add topics in **Settings → General → Topics**:
```
machine-learning, fairness, computer-vision, nlp, counterfactual, bias-detection, 
responsible-ai, diffusion-models, multimodal, python, pytorch, research-tool
```

#### Security Settings
Navigate to **Settings → Security & analysis** and enable:
- Dependency graph: ✓
- Dependabot alerts: ✓
- Dependabot security updates: ✓
- Secret scanning: ✓
- Push protection: ✓

### 3. Required Secrets

Add these secrets in **Settings → Secrets and variables → Actions**:

```bash
# Required for releases
GITHUB_TOKEN  # Automatically provided by GitHub

# Optional: For enhanced security scanning
SNYK_TOKEN    # From snyk.io
SONAR_TOKEN   # From sonarcloud.io

# Optional: For deployment
DOCKER_USERNAME
DOCKER_PASSWORD
```

### 4. Third-party Integrations

#### CodeQL Analysis
Enable GitHub Advanced Security in repository settings if you have access.

#### Dependabot Configuration
The `dependabot.yml` file is already configured for:
- Python dependencies (daily updates)
- GitHub Actions (weekly updates)
- Docker (weekly updates)

#### Pre-commit Hooks
Install pre-commit hooks locally:
```bash
pip install pre-commit
pre-commit install
```

### 5. Monitoring Setup

#### GitHub Discussions
Enable Discussions in **Settings → General → Features → Discussions**

#### Issue Templates
Issue templates are already configured in `.github/ISSUE_TEMPLATE/`

#### Pull Request Template
PR template is configured in `.github/PULL_REQUEST_TEMPLATE.md`

### 6. Documentation Website (Optional)

To enable documentation hosting:

1. **GitHub Pages**: Go to **Settings → Pages** and set source to "GitHub Actions"
2. **ReadTheDocs**: Connect repository at readthedocs.org
3. **MkDocs**: The `mkdocs.yml` is already configured

### 7. Container Registry (Optional)

For Docker image publishing:

1. **GitHub Container Registry**: 
   - Enable in package settings
   - Configure workflow permissions

2. **Docker Hub**:
   - Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets
   - Update workflow configuration

### 8. Performance Monitoring

#### Metrics Collection
The metrics collection script is available at `scripts/collect_metrics.py`. 

To automate collection, add to your CI workflow:
```yaml
- name: Collect Metrics
  run: python scripts/collect_metrics.py --save
```

#### Repository Insights
Enable repository insights in **Settings → General → Features → Insights**

### 9. Community Health

#### Community Profile
Check your community profile at: `https://github.com/yourusername/multimodal-counterfactual-lab/community`

Ensure all items are complete:
- [x] Description
- [x] README
- [x] Code of conduct
- [x] Contributing guidelines
- [x] License
- [x] Security policy
- [x] Issue templates
- [x] Pull request template

### 10. Final Verification

After completing manual setup:

1. **Test workflows**: Create a test PR to verify CI/CD
2. **Security scan**: Check security tab for any issues
3. **Metrics collection**: Run `python scripts/collect_metrics.py`
4. **Documentation**: Verify docs are building correctly

## Validation Checklist

- [ ] Branch protection rules configured
- [ ] Workflows copied and enabled
- [ ] Security features enabled
- [ ] Secrets configured (if needed)
- [ ] Dependabot enabled
- [ ] Pre-commit hooks installed
- [ ] Documentation site enabled
- [ ] Community profile complete
- [ ] Test CI/CD pipeline
- [ ] Verify metrics collection

## Support

If you need help with any of these setup steps:
- Check the [GitHub documentation](https://docs.github.com)
- Open a discussion in the repository
- Contact the maintainers

---

**Note**: This setup ensures your repository follows security and development best practices while maintaining compliance with organizational policies.