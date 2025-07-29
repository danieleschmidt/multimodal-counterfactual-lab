# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup with comprehensive SDLC framework
- GitHub issue and PR templates for standardized contributions
- Docker containerization with multi-stage builds
- Comprehensive security scanning with multiple tools
- Dependency security monitoring with Dependabot
- Software Bill of Materials (SBOM) generation
- Performance benchmarking infrastructure
- Enhanced testing framework with multiple test types
- Automated dependency updates and vulnerability scanning

### Changed
- Enhanced development workflow with additional quality gates
- Improved documentation structure and automation

### Security
- Added comprehensive security scanning pipeline
- Implemented secret detection and prevention
- Enhanced container security with Trivy scanning
- Added license compliance monitoring

## [0.1.0] - 2025-01-XX

### Added
- Initial release of Multimodal Counterfactual Lab
- MoDiCF (Diffusion-based Multimodal Counterfactual Generation) pipeline  
- ICG (Interpretable Counterfactual Generation) generator
- Skew-aware sampling for balanced counterfactual generation
- CITS (Counterfactual Image-Text Score) evaluation metrics
- Bias evaluation framework with multiple fairness metrics
- Command-line interface for batch processing
- Web interface built with Streamlit
- Comprehensive documentation and examples
- Support for multiple VLM architectures (CLIP, ALIGN, FLAVA)
- Integration with popular ML frameworks (PyTorch, Transformers, Diffusers)

### Security
- Input validation for all user-provided data
- Secure model loading and inference pipelines
- Privacy-preserving counterfactual generation
- Audit logging for compliance requirements

---

## Release Notes Template

When creating a new release, use the following template:

### Version X.Y.Z - YYYY-MM-DD

**🚀 New Features**
- List new features here

**🐛 Bug Fixes** 
- List bug fixes here

**📚 Documentation**
- List documentation updates here

**🔒 Security**
- List security improvements here

**⚡ Performance**
- List performance improvements here

**💥 Breaking Changes**
- List any breaking changes here with migration guide

**🧪 For Developers**
- List changes relevant to contributors/developers

**📦 Dependencies**
- Notable dependency updates

**🙏 Contributors**
- Thank contributors to this release