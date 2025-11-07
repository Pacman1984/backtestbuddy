# Test Suite for backtestbuddy

This directory contains the test suite for the backtestbuddy package, organized using a **mirror strategy** that reflects the source code structure.

## Directory Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Shared fixtures and pytest configuration
├── README.md                       # This file
├── unit/                           # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── test_sport_backtest.py  # Unit tests for backtest module
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── test_sport_metrics.py   # Unit tests for metrics module
│   │   └── test_risk_adjusted_roi.py  # Focused tests for risk-adjusted ROI
│   ├── plots/
│   │   ├── __init__.py
│   │   └── test_sport_plots.py     # Unit tests for plots module
│   └── strategies/
│       ├── __init__.py
│       └── test_sport_strategies.py  # Unit tests for strategies module
└── integration/                    # Integration tests (slower, multi-component)
    ├── __init__.py
    └── test_backtest_workflow.py  # End-to-end backtest workflow tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests that verify individual components work correctly:
- **Focus**: Single functions, classes, or methods
- **Speed**: Fast (< 1 second per test)
- **Dependencies**: Minimal external dependencies
- **Isolation**: No interaction with external systems
- **Markers**: Automatically marked with `@pytest.mark.unit`

**Example**: Testing that `calculate_roi()` returns correct value for known input.

### Integration Tests (`tests/integration/`)
Tests that verify multiple components work together correctly:
- **Focus**: Component interactions and workflows
- **Speed**: Slower (may take several seconds)
- **Dependencies**: Multiple backtestbuddy components
- **Scope**: End-to-end scenarios
- **Markers**: Automatically marked with `@pytest.mark.integration`

**Example**: Testing complete backtest workflow from data input to results generation.

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Only Unit Tests
```bash
pytest -m unit
```

### Run Only Integration Tests
```bash
pytest -m integration
```

### Run Tests for Specific Module
```bash
# Run all metrics tests
pytest -m metrics

# Run all strategy tests
pytest -m strategies

# Run specific test file
pytest tests/unit/metrics/test_sport_metrics.py
```

### Run Tests with Coverage
```bash
pytest --cov=src/backtestbuddy --cov-report=html
```

### Run Tests in Verbose Mode
```bash
pytest -v
```

### Run Tests and Stop on First Failure
```bash
pytest -x
```

## Test Markers

Tests are automatically marked based on their location. Available markers:

- `@pytest.mark.unit` - Unit tests (automatically applied to `tests/unit/*`)
- `@pytest.mark.integration` - Integration tests (automatically applied to `tests/integration/*`)
- `@pytest.mark.metrics` - Metrics-related tests
- `@pytest.mark.strategies` - Strategy-related tests
- `@pytest.mark.backtest` - Backtest-related tests
- `@pytest.mark.plots` - Plot-related tests
- `@pytest.mark.slow` - Tests that take a long time to run (manually applied)

## Writing New Tests

### Adding Unit Tests

1. Create test file in appropriate `tests/unit/<module>/` directory
2. Follow naming convention: `test_<module_name>.py`
3. Use deterministic test data (avoid randomness)
4. Keep tests focused and isolated
5. Test one thing per test function

**Example**:
```python
# tests/unit/metrics/test_new_metric.py
"""Unit tests for new_metric function."""
import pytest
from backtestbuddy.metrics.sport_metrics import new_metric

class TestNewMetric:
    def test_new_metric_with_known_values(self):
        """Test new_metric returns expected value for known input."""
        data = create_test_data()
        result = new_metric(data)
        assert result == pytest.approx(expected_value)
```

### Adding Integration Tests

1. Create test file in `tests/integration/` directory
2. Test complete workflows with multiple components
3. Verify end-to-end behavior
4. Use realistic data and scenarios

**Example**:
```python
# tests/integration/test_new_workflow.py
"""Integration tests for new workflow."""
def test_complete_workflow():
    """Test complete workflow from start to finish."""
    backtest = create_backtest()
    backtest.run()
    results = backtest.get_detailed_results()
    metrics = backtest.calculate_metrics()
    # Verify the complete pipeline works
    assert results is not None
    assert metrics['ROI [%]'] > 0
```

## Shared Fixtures

Common fixtures are defined in `tests/conftest.py` and available to all tests:

- `sample_deterministic_data`: Standard deterministic test data
- `dummy_classifier`: Sklearn dummy classifier with fixed seed
- `sample_backtest_results`: Sample backtest results DataFrame

Use these fixtures to maintain consistency across tests.

## Test Quality Guidelines

### ✅ Good Practices
- Use deterministic test data (no random values)
- Test one behavior per test function
- Use descriptive test names that explain what is being tested
- Verify exact expected values when possible
- Test edge cases and boundary conditions
- Keep tests independent (no shared state between tests)

### ❌ Anti-patterns to Avoid
- Using random data that makes tests non-deterministic
- Testing implementation details instead of behavior
- Mocking the code under test (mock external dependencies only)
- Overly broad assertions (`assert x is not None`)
- Tests that depend on execution order
- Copy-pasting algorithm logic into test expectations

## Continuous Integration

Tests are automatically run on:
- Every push to repository
- Every pull request
- Before releases

All tests must pass before code can be merged.

## Test Coverage

We aim for:
- **Unit tests**: >90% code coverage
- **Integration tests**: Cover all critical workflows
- **Overall**: >85% combined coverage

Check current coverage:
```bash
pytest --cov=src/backtestbuddy --cov-report=term
```

## Questions or Issues?

If you have questions about the test structure or find issues:
1. Check this README first
2. Look at existing tests for examples
3. Refer to pytest documentation: https://docs.pytest.org

