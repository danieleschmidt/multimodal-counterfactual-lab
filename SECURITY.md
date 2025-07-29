# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly:

### How to Report

**Please do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security issues by emailing: **security@terragon.ai**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential impact assessment
- Suggested fixes (if any)

### Response Timeline

- **Initial Response**: Within 24 hours
- **Status Update**: Within 72 hours
- **Fix Timeline**: Critical issues within 7 days, others within 30 days

### Scope

This security policy covers:
- The core `counterfactual_lab` package
- CLI tools and web interfaces
- Model inference pipelines
- Data processing components

### Out of Scope

- Vulnerabilities in third-party dependencies (report to respective maintainers)
- Issues in user-provided models or datasets
- Social engineering attacks
- Physical security issues

## Security Best Practices

When using this library:

1. **Model Security**: Only use trusted pre-trained models
2. **Data Privacy**: Ensure compliance with data protection regulations
3. **Environment Security**: Use virtual environments and dependency pinning
4. **Access Control**: Implement proper authentication for web interfaces
5. **Audit Trails**: Enable logging for bias evaluation results

## Security Features

- Input validation for all user-provided data
- Secure model loading and inference
- Privacy-preserving counterfactual generation
- Audit logging for compliance requirements
- Dependency vulnerability scanning

## Contact

For security-related questions: security@terragon.ai
For general inquiries: daniel@terragon.ai