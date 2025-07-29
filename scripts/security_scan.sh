#!/bin/bash
# Security scanning script for Multimodal Counterfactual Lab

set -e

echo "🔒 Running comprehensive security scans..."

# Create reports directory
mkdir -p security_reports

# 1. Dependency vulnerability scanning with Safety
echo "📦 Scanning dependencies with Safety..."
pip install safety
safety check --json --output security_reports/safety_report.json || true
safety check --short-report || true

# 2. Code security analysis with Bandit
echo "🔍 Running Bandit security analysis..."
bandit -r src/ -f json -o security_reports/bandit_report.json || true
bandit -r src/ --severity-level medium || true

# 3. Secret detection with detect-secrets
echo "🔐 Scanning for secrets..."
pip install detect-secrets
detect-secrets scan --all-files --force-use-all-plugins > security_reports/secrets_baseline.json || true

# 4. License compliance check
echo "📄 Checking license compliance..."
pip install pip-licenses
pip-licenses --format=json --output-file=security_reports/licenses.json
pip-licenses --summary

# 5. Docker image scanning (if Docker available)
if command -v docker &> /dev/null; then
    echo "🐳 Scanning Docker image with Trivy..."
    # Install Trivy
    curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    
    # Build and scan image
    docker build -t counterfactual-lab:security-scan .
    trivy image --format json --output security_reports/trivy_report.json counterfactual-lab:security-scan || true
    trivy image --severity HIGH,CRITICAL counterfactual-lab:security-scan || true
else
    echo "⚠️  Docker not available, skipping container security scan"
fi

# 6. SBOM generation
echo "📋 Generating Software Bill of Materials (SBOM)..."
pip install cyclonedx-bom
cyclonedx-py -o security_reports/sbom.json

# 7. Audit Python packages
echo "🔎 Auditing Python packages..."
pip install pip-audit
pip-audit --format=json --output=security_reports/pip_audit.json || true
pip-audit --desc || true

# 8. Generate security summary
echo "📊 Generating security summary..."
cat > security_reports/README.md << 'EOF'
# Security Scan Results

This directory contains security scan results from various tools:

## Reports Generated

- `safety_report.json` - Dependency vulnerability scan (Safety)
- `bandit_report.json` - Static code security analysis (Bandit)  
- `secrets_baseline.json` - Secret detection baseline (detect-secrets)
- `licenses.json` - License compliance report (pip-licenses)
- `trivy_report.json` - Container vulnerability scan (Trivy)
- `sbom.json` - Software Bill of Materials (CycloneDX)
- `pip_audit.json` - Python package audit (pip-audit)

## Remediation

Review each report and address any HIGH or CRITICAL severity findings:

1. **Dependencies**: Update vulnerable packages to secure versions
2. **Code**: Fix security issues identified by Bandit
3. **Secrets**: Remove any detected secrets and rotate credentials
4. **Licenses**: Ensure license compliance for all dependencies
5. **Containers**: Update base images and dependencies

## Automation

These scans are run automatically in CI/CD pipelines and should be reviewed before releases.
EOF

echo "✅ Security scanning complete! Check security_reports/ directory for results."
echo "🚨 Address any HIGH or CRITICAL findings before deployment."