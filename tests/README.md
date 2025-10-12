# Dialectus Engine Tests

This directory contains the test suite for the Dialectus Engine library.

## Structure

```
tests/
├── __init__.py                  # Makes tests a package
├── conftest.py                  # Shared fixtures and pytest configuration
├── test_format_registry.py      # Tests for format registry
└── README.md                    # This file
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run tests in a specific file
pytest tests/test_format_registry.py

# Run a specific test function
pytest tests/test_format_registry.py::test_registry_initialization

# Run tests in a class
pytest tests/test_format_registry.py::TestFormatRetrieval
```

### Useful Options

```bash
# Stop at first failure
pytest -x

# Show print statements
pytest -s

# Run tests matching a keyword
pytest -k "oxford"

# Run only fast tests (exclude slow)
pytest -m "not slow"

# Show local variables in tracebacks
pytest -l

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Coverage

```bash
# Run with coverage report
pytest --cov=dialectus.engine

# Generate HTML coverage report
pytest --cov=dialectus.engine --cov-report=html

# View coverage for specific module
pytest --cov=dialectus.engine.formats.registry tests/test_format_registry.py
```

## Test Organization

### Test File Naming
- Test files: `test_*.py` or `*_test.py`
- Test functions: `test_*()`
- Test classes: `Test*`

### Test Categories

Tests can be marked with decorators:

```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass
```

Run specific categories:
```bash
pytest -m unit          # Run only unit tests
pytest -m "not slow"    # Skip slow tests
```

## Writing Tests

### Basic Test Structure

```python
def test_something():
    # Arrange - Set up test data
    registry = FormatRegistry()

    # Act - Perform the action
    result = registry.list_formats()

    # Assert - Check the result
    assert len(result) > 0
```

### Using Fixtures

```python
def test_with_fixture(registry):  # 'registry' fixture auto-injected
    formats = registry.list_formats()
    assert "oxford" in formats
```

### Parametrized Tests

```python
@pytest.mark.parametrize("format_name", ["oxford", "parliamentary"])
def test_all_formats(format_name, registry):
    debate_format = registry.get_format(format_name)
    assert debate_format.name == format_name
```

### Testing Exceptions

```python
def test_error_handling():
    registry = FormatRegistry()

    with pytest.raises(ValueError, match="Unknown format"):
        registry.get_format("nonexistent")
```

## Key Patterns Demonstrated

The `test_format_registry.py` file demonstrates:

1. **Fixtures** - Reusable test components (`@pytest.fixture`)
2. **Parametrized tests** - Testing multiple inputs efficiently
3. **Exception testing** - Using `pytest.raises()`
4. **Test classes** - Organizing related tests
5. **Mock objects** - Creating test doubles
6. **Integration tests** - Testing complete workflows
7. **Edge case testing** - Boundary conditions

## Adding New Tests

When adding tests for new modules:

1. Create a new file: `tests/test_<module_name>.py`
2. Import the module you're testing
3. Create fixtures in `conftest.py` if needed across files
4. Follow the existing patterns in `test_format_registry.py`

### Example: Testing Cache Manager

```python
# tests/test_cache_manager.py
import pytest
from dialectus.engine.models.cache_manager import ModelCacheManager

@pytest.fixture
def cache_manager(tmp_path):
    """Provide a cache manager with temporary directory."""
    return ModelCacheManager(cache_dir=tmp_path / "cache")

def test_cache_set_and_get(cache_manager):
    cache_manager.set("provider", "endpoint", {"data": "value"})
    result = cache_manager.get("provider", "endpoint")
    assert result == {"data": "value"}
```

## Best Practices

1. **Test one thing** - Each test should verify one behavior
2. **Clear names** - Test names should describe what they test
3. **Arrange-Act-Assert** - Structure tests clearly
4. **Independent tests** - Tests shouldn't depend on each other
5. **Use fixtures** - Share setup code via fixtures
6. **Test edge cases** - Empty inputs, null values, boundaries
7. **Test errors** - Verify error handling works correctly

## CI Integration

Tests run automatically in GitHub Actions on:
- Push to `main`
- Pull requests

See `.github/workflows/ci.yml` for CI configuration.

## Learning Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Real Python - Pytest Tutorial](https://realpython.com/pytest-python-testing/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Parametrize](https://docs.pytest.org/en/stable/parametrize.html)

## Next Steps

Good candidates for additional tests:

1. **Cache Manager** - File I/O, TTL, expiry logic
2. **Config Models** - Pydantic validation, field validators
3. **Debate Models** - Dataclass behavior, defaults
4. **Individual Formats** - Phase generation, side labels
5. **Judge Factory** - Factory pattern, dependency injection

Start with modules that have:
- Pure Python logic (no I/O)
- Clear inputs/outputs
- Minimal dependencies
