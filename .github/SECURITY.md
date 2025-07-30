# Security Policy

## Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability in the Multimodal Counterfactual Lab, please report it responsibly:

### 🚨 Critical Vulnerabilities
- **Contact**: security@terragon.ai
- **Response Time**: Within 24 hours
- **Expected Resolution**: 1-7 days

### 📧 Reporting Process
1. **Email**: Send details to security@terragon.ai with subject "SECURITY: [Brief Description]"
2. **Include**: 
   - Detailed description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes

### 🔒 What We Consider Security Issues
- Remote code execution vulnerabilities
- Privilege escalation issues
- Information disclosure vulnerabilities
- Authentication bypass
- Model poisoning attacks
- Adversarial input handling flaws
- Data leakage in counterfactual generation
- Bias amplification vulnerabilities

## Security Measures

### 🛡️ Current Protections
- **Dependency Scanning**: Automated vulnerability detection with Safety and pip-audit
- **Static Analysis**: CodeQL security analysis on all code changes
- **Container Security**: Trivy scanning for Docker images
- **Secret Detection**: TruffleHog scans for exposed credentials
- **Input Validation**: Robust validation for all user inputs
- **Model Security**: Safe loading and execution of ML models

### 🔐 Security Best Practices for Contributors
- Never commit secrets, API keys, or credentials
- Use parameterized queries and input validation
- Follow secure coding guidelines in our CONTRIBUTING.md
- Validate all external inputs including image and text data
- Use secure defaults for all configuration options

### 📊 Security Compliance
This project follows:
- **OWASP Top 10** security guidelines
- **CIS Security Benchmarks** for containerized deployments
- **NIST Cybersecurity Framework** principles
- **OpenSSF Scorecard** recommendations

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ Active support  |
| < 0.1   | ❌ No support      |

## Security Updates
- Security patches are released immediately upon discovery
- Critical updates bypass normal release cycles
- All security fixes are documented in CHANGELOG.md
- Security advisories published through GitHub Security tab

## Response Timeline
- **Critical**: 24 hours
- **High**: 72 hours  
- **Medium**: 1 week
- **Low**: Next release cycle

Thank you for helping keep Multimodal Counterfactual Lab secure! 🔒