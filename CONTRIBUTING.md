# Contributing to Multimodal Counterfactual Lab

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/multimodal-counterfactual-lab.git
   cd multimodal-counterfactual-lab
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   make install-dev
   ```

3. **Run tests to verify setup**
   ```bash
   make test
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   make lint          # Check code style
   make type-check    # Run type checking
   make test-cov      # Run tests with coverage
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new counterfactual method"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **mypy**: Type checking
- **pytest**: Testing

All checks are enforced by pre-commit hooks and CI.

## Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Place tests in appropriate directories:
  - `tests/unit/`: Fast, isolated tests
  - `tests/integration/`: Tests that require multiple components
  - `tests/e2e/`: End-to-end workflow tests

## Documentation

- Update docstrings for new/modified functions
- Add examples to docstrings when helpful
- Update the changelog for significant changes
- Consider adding tutorials for major features

## Pull Request Guidelines

1. **Title**: Use conventional commits format
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for adding tests

2. **Description**: Include:
   - Summary of changes
   - Motivation for the change
   - Any breaking changes
   - Testing performed

3. **Checklist**:
   - [ ] Tests pass (`make test`)
   - [ ] Code is formatted (`make format`)
   - [ ] Type checking passes (`make type-check`)
   - [ ] Documentation updated
   - [ ] Changelog updated (if significant)

## Priority Contribution Areas

We especially welcome contributions in:

1. **New Generation Methods**: Additional counterfactual generation techniques
2. **Fairness Metrics**: Implementation of bias evaluation metrics
3. **Model Integrations**: Support for more VLM architectures
4. **Documentation**: Tutorials, examples, and guides
5. **Performance**: Optimization and efficiency improvements

## Bug Reports

Use GitHub Issues to report bugs:

1. **Search existing issues** to avoid duplicates
2. **Use the bug report template**
3. **Include minimal reproduction steps**
4. **Specify your environment** (Python version, OS, dependencies)

## Feature Requests

For new features:

1. **Check existing issues** and discussions
2. **Use the feature request template**
3. **Explain the use case and benefit**
4. **Consider proposing an implementation approach**

## Questions and Support

- **GitHub Discussions**: For questions and general discussion
- **Discord**: Join our community server (link in README)
- **Documentation**: Check existing docs before asking

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). 
Please read and follow it in all your interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.