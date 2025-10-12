"""Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Shared fixtures available to all test modules
- Pytest configuration hooks
- Common test utilities

Learning notes:
- conftest.py is a special filename recognized by pytest
- Fixtures defined here are available to all tests without importing
- This is where you put fixtures used across multiple test files
"""

import pytest


# =============================================================================
# SHARED FIXTURES - Available to all test modules
# =============================================================================


@pytest.fixture
def sample_debate_topic() -> str:
    """Provide a standard debate topic for testing.

    Learning notes:
    - Simple fixture returning a string
    - Can be used in any test by adding 'sample_debate_topic' parameter
    - With strict pyright, fixtures need return type annotations
    """
    return "Should artificial intelligence be regulated by governments?"


@pytest.fixture
def two_participants() -> list[str]:
    """Provide two participant IDs for standard debates.

    Learning notes:
    - Fixtures can return any Python object
    - List fixtures are commonly used for test data
    - Use modern type syntax: list[str] not List[str]
    """
    return ["model_a", "model_b"]


@pytest.fixture
def four_participants() -> list[str]:
    """Provide four participant IDs for larger debates."""
    return ["model_a", "model_b", "model_c", "model_d"]


# =============================================================================
# PYTEST CONFIGURATION HOOKS
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers.

    Learning notes:
    - Hooks allow customizing pytest behavior
    - Markers are used to categorize tests
    - Use with @pytest.mark.slow, etc.
    - Need to import pytest.Config type for strict mode
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# =============================================================================
# FUTURE: Add more shared fixtures as needed
# =============================================================================

# Examples of fixtures you might add later:
#
# @pytest.fixture
# def temp_cache_dir(tmp_path):
#     """Provide a temporary directory for cache testing."""
#     cache_dir = tmp_path / "cache"
#     cache_dir.mkdir()
#     return cache_dir
#
# @pytest.fixture
# def mock_model_config():
#     """Provide a mock ModelConfig for testing."""
#     from dialectus.engine.config.settings import ModelConfig
#     return ModelConfig(
#         name="test-model",
#         provider="ollama",
#         personality="neutral",
#         max_tokens=100,
#         temperature=0.7
#     )
