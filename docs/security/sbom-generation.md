# Software Bill of Materials (SBOM) Generation

## Overview

This document outlines SBOM generation procedures for supply chain security and compliance with executive orders and regulations requiring software transparency.

## SBOM Standards Supported

- **SPDX 2.3+**: Industry standard format
- **CycloneDX**: OWASP standard for security use cases
- **SWID**: ISO/IEC 19770-2 software identification

## Generation Methods

### 1. Automated Generation (Recommended)

```bash
# Install SBOM tools
pip install cyclonedx-bom sbom-tool

# Generate CycloneDX SBOM
cyclonedx-py -o sbom-cyclonedx.json

# Generate SPDX SBOM  
python -m pip_audit --format=json --output=sbom-spdx.json

# Generate comprehensive SBOM with dependencies
syft packages dir:. -o json=sbom-syft.json
```

### 2. CI/CD Integration

Add to `.github/workflows/security.yml`:

```yaml
- name: Generate SBOM
  uses: anchore/sbom-action@v0
  with:
    path: ./
    format: spdx-json
    upload-artifact: true
    upload-release-assets: true
```

### 3. Docker Image SBOM

```bash
# Generate SBOM for container images
syft multimodal-counterfactual-lab:latest -o json=container-sbom.json

# Scan container SBOM for vulnerabilities
grype sbom:container-sbom.json
```

## SBOM Validation

### Required Components

- [x] Package name and version
- [x] Supplier information
- [x] License information
- [x] Cryptographic hashes
- [x] Dependency relationships
- [x] Vulnerability data integration

### Validation Commands

```bash
# Validate SPDX SBOM
spdx-tools-python verify sbom-spdx.json

# Validate CycloneDX SBOM
cyclonedx validate --input-file sbom-cyclonedx.json
```

## Compliance Integration

### Regulatory Requirements

- **EU Cyber Resilience Act**: SBOM mandatory for CE marking
- **US Executive Order 14028**: Federal software procurement
- **NIST SSDF**: Secure software development framework

### Audit Trail

```bash
# Generate audit-ready SBOM with signatures
cosign sign-blob --key=private.key sbom-spdx.json
cosign verify-blob --key=public.key --signature=sbom-spdx.json.sig sbom-spdx.json
```

## Vulnerability Correlation

### Integration with Security Scanners

```bash
# Correlate SBOM with CVE databases
grype sbom:sbom-spdx.json --output json

# Generate vulnerability report
trivy sbom sbom-cyclonedx.json --format json
```

### Automated Alerts

Configure alerts for:
- New vulnerabilities in SBOM components
- License compliance violations
- Supply chain tampering detection
- Outdated dependency identification

## Storage and Distribution

### Secure Storage
- Store SBOMs in secure artifact registry
- Implement access controls and audit logging
- Maintain historical SBOM versions
- Encrypt SBOMs for sensitive applications

### Distribution Channels
- Attach to software releases
- Publish to transparency logs
- Share with customers and auditors
- Integrate with procurement systems

## Automation Scripts

Reference implementation in `scripts/generate_sbom.py`:

```python
#!/usr/bin/env python3
"""
Automated SBOM generation with multiple format support
"""
import subprocess
import json
from pathlib import Path

class SBOMGenerator:
    def __init__(self, project_path: Path):
        self.project_path = project_path
        
    def generate_all_formats(self):
        \"\"\"Generate SBOMs in all supported formats\"\"\"
        formats = ['spdx-json', 'cyclonedx-json', 'syft-json']
        for fmt in formats:
            self.generate_sbom(fmt)
            
    def generate_sbom(self, format_type: str):
        \"\"\"Generate SBOM in specified format\"\"\"
        # Implementation details in actual script
        pass
        
    def validate_sbom(self, sbom_path: Path):
        \"\"\"Validate generated SBOM\"\"\"
        # Validation logic
        pass
```

## Best Practices

1. **Regular Generation**: Generate SBOMs on every release
2. **Format Diversity**: Support multiple SBOM formats
3. **Signature Verification**: Sign SBOMs for integrity
4. **Vulnerability Integration**: Correlate with security scanners
5. **Compliance Documentation**: Maintain audit trails

## References

- [NTIA SBOM Standards](https://www.ntia.gov/SBOM)
- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Standard](https://cyclonedx.org/)
- [CISA SBOM Guide](https://www.cisa.gov/sbom)