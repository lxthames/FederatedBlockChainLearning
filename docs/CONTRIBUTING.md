# Contributing Guidelines

Thank you for your interest in contributing to the Blockchain-Based Decentralized Federated Learning Framework!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Testing](#testing)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information

### Suggesting Features

1. Check if the feature has been suggested
2. Create a new issue with:
   - Clear description
   - Use case and motivation
   - Proposed implementation (if any)

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/Federated_BL.git
cd Federated_BL
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Dependencies

```bash
pip install pytest black flake8
```

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions/classes
- Keep functions focused and small

### Code Formatting

```bash
# Format code with black
black Reduced_Layers_MUltihead.ipynb

# Check with flake8
flake8 src/
```

### Documentation

- Write clear docstrings
- Update relevant documentation
- Add comments for complex logic
- Keep README updated

### Example

```python
def federated_round_async(
    self, 
    local_epochs: int = 1, 
    batch_size: int = 128,
    lr: float = 0.001, 
    max_staleness: int = 3
) -> Dict:
    """
    Execute one federated learning round with PoP/PoC validation.
    
    Args:
        local_epochs: Number of local training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_staleness: Maximum allowed model staleness
        
    Returns:
        Dictionary with round statistics
    """
    # Implementation
    pass
```

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Update relevant docs
2. **Add Tests**: Include tests for new features
3. **Check Style**: Run formatters and linters
4. **Test Locally**: Ensure everything works

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No merge conflicts
- [ ] Clear commit messages

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add custom PoC threshold configuration
fix: Resolve blockchain persistence error
docs: Update API documentation
refactor: Simplify model aggregation logic
test: Add tests for PoP registration
```

## Documentation

### Documentation Structure

- **Architecture Docs**: System design and architecture
- **API Docs**: Complete API reference
- **Guides**: User guides and tutorials
- **Code Comments**: Inline documentation

### Writing Documentation

- Use clear, concise language
- Include examples
- Add diagrams where helpful
- Keep up-to-date with code

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_blockchain.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Test all new features
- Include edge cases
- Test error handling
- Maintain test coverage

### Example Test

```python
def test_poc_evaluation():
    """Test PoC contribution evaluation."""
    poc = ProofOfContribution(validation_data=(X_val, Y_val))
    contribution = poc.evaluate_contribution(...)
    assert contribution['overall_score'] >= 0
    assert contribution['overall_score'] <= 1
```

## Areas for Contribution

### High Priority

- [ ] Distributed P2P networking
- [ ] Database-backed persistence
- [ ] Public key cryptography
- [ ] Comprehensive test suite
- [ ] Performance optimizations

### Medium Priority

- [ ] Additional consensus mechanisms
- [ ] Advanced PoP methods
- [ ] Enhanced PoC metrics
- [ ] Model compression
- [ ] Monitoring and logging

### Low Priority

- [ ] UI/Web interface
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Performance benchmarks
- [ ] Additional examples

## Questions?

- Open an issue for questions
- Check existing documentation
- Review closed issues/PRs

## Thank You!

Your contributions make this project better. Thank you for taking the time to contribute!

