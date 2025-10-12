"""Tests for the debate format registry."""

import pytest

from dialectus.engine.debate_engine.types import Position
from dialectus.engine.formats.base import DebateFormat, FormatPhase
from dialectus.engine.formats.registry import FormatRegistry

# =============================================================================
# FIXTURES - Reusable test components
# =============================================================================


@pytest.fixture
def registry() -> FormatRegistry:
    """Provide a fresh FormatRegistry instance for each test.

    This fixture ensures test isolation - each test gets its own registry
    instance, preventing tests from interfering with each other.

    Learning notes (strict typing):
    - Fixtures must have return type annotations in strict mode
    - This helps pyright understand what fixtures provide to tests
    """
    return FormatRegistry()


@pytest.fixture
def sample_participants() -> list[str]:
    """Provide a standard list of participants for testing.

    Learning notes (strict typing):
    - Use modern list[str] syntax, not List[str]
    """
    return ["model_a", "model_b"]


# =============================================================================
# BASIC TESTS - Simple assertions and behavior verification
# =============================================================================


def test_registry_initialization(registry: FormatRegistry) -> None:
    """Test that a new registry initializes with built-in formats.

    Learning notes:
    - Basic fixture usage (registry parameter)
    - Simple assertions with assert
    - Testing that collections are not empty
    - In strict mode: test functions must be annotated with -> None
    - In strict mode: fixture parameters must have type annotations
    """
    formats = registry.list_formats()
    assert len(formats) > 0, "Registry should contain built-in formats"
    assert isinstance(formats, list), "list_formats() should return a list"


def test_built_in_formats_registered(registry: FormatRegistry) -> None:
    """Test that all expected built-in formats are registered.

    Learning notes:
    - Testing for specific values in collections
    - Multiple assertions in one test
    - Using 'in' operator for membership testing
    """
    formats = registry.list_formats()

    # Check that all expected formats are present
    assert "oxford" in formats
    assert "parliamentary" in formats
    assert "socratic" in formats
    assert "public_forum" in formats


def test_list_formats_returns_format_names(registry: FormatRegistry) -> None:
    """Test that list_formats returns the correct format names."""
    formats = registry.list_formats()

    # All items should be strings
    for format_name in formats:
        assert isinstance(format_name, str)

    # Should have at least the 4 built-in formats
    assert len(formats) >= 4


# =============================================================================
# FORMAT RETRIEVAL TESTS
# =============================================================================


def test_get_format_returns_format_instance(registry: FormatRegistry) -> None:
    """Test that get_format returns a DebateFormat instance.

    Learning notes:
    - Testing return types with isinstance()
    - Verifying interface compliance
    """
    oxford_format = registry.get_format("oxford")

    assert isinstance(oxford_format, DebateFormat)
    assert oxford_format.name == "oxford"


def test_get_format_returns_new_instance_each_time(registry: FormatRegistry) -> None:
    """Test that get_format creates a new instance on each call.

    Learning notes:
    - Testing object identity with 'is not'
    - Ensuring factory pattern behavior
    """
    format_1 = registry.get_format("oxford")
    format_2 = registry.get_format("oxford")

    # Should be different instances
    assert format_1 is not format_2

    # But same type and name
    assert type(format_1) is type(format_2)
    assert format_1.name == format_2.name


# =============================================================================
# PARAMETRIZED TESTS - Testing multiple inputs efficiently
# =============================================================================


@pytest.mark.parametrize(
    "format_name,expected_display_name",
    [
        ("oxford", "Oxford"),
        ("parliamentary", "Parliamentary"),
        ("socratic", "Socratic"),
        ("public_forum", "Public Forum"),
    ],
)
def test_format_display_names(
    registry: FormatRegistry, format_name: str, expected_display_name: str
) -> None:
    """Test that each format has the correct display name.

    Learning notes:
    - @pytest.mark.parametrize decorator for running same test with different inputs
    - First arg is parameter names (comma-separated string)
    - Second arg is list of tuples with test values
    - This runs 4 separate tests (one per tuple)
    - Parametrized values need type annotations too in strict mode
    """
    debate_format = registry.get_format(format_name)
    assert debate_format.display_name == expected_display_name


@pytest.mark.parametrize(
    "format_name", ["oxford", "parliamentary", "socratic", "public_forum"]
)
def test_all_formats_have_descriptions(
    registry: FormatRegistry, format_name: str
) -> None:
    """Test that all formats have non-empty descriptions.

    Learning notes:
    - Parametrize with single parameter
    - Testing that strings are not empty
    """
    debate_format = registry.get_format(format_name)

    assert debate_format.description
    assert len(debate_format.description) > 0
    assert isinstance(debate_format.description, str)


@pytest.mark.parametrize(
    "format_name", ["oxford", "parliamentary", "socratic", "public_forum"]
)
def test_all_formats_return_phases(
    registry: FormatRegistry, format_name: str, sample_participants: list[str]
) -> None:
    """Test that all formats can generate phases.

    Learning notes:
    - Using multiple fixtures (registry + sample_participants)
    - Testing that methods return expected structure
    """
    debate_format = registry.get_format(format_name)
    phases = debate_format.get_phases(sample_participants)

    assert isinstance(phases, list)
    assert len(phases) > 0, f"{format_name} should have at least one phase"


# =============================================================================
# EXCEPTION TESTING - Verifying error handling
# =============================================================================


def test_get_format_raises_error_for_unknown_format(registry: FormatRegistry) -> None:
    """Test that requesting an unknown format raises ValueError.

    Learning notes:
    - pytest.raises context manager for testing exceptions
    - Verifying exception type
    - Checking exception message content with 'match' parameter
    """
    with pytest.raises(ValueError, match="Unknown format: nonexistent"):
        registry.get_format("nonexistent")


def test_error_message_lists_available_formats(registry: FormatRegistry) -> None:
    """Test that error message includes available formats.

    Learning notes:
    - Capturing exception for detailed inspection
    - Accessing exception message with str(exc_info.value)
    """
    with pytest.raises(ValueError) as exc_info:
        registry.get_format("invalid_format")

    error_message = str(exc_info.value)
    assert "Unknown format: invalid_format" in error_message
    assert "Available:" in error_message

    # Check that at least some format names are in the error
    assert "oxford" in error_message or "parliamentary" in error_message


# =============================================================================
# FORMAT DESCRIPTIONS TESTS
# =============================================================================


def test_get_format_descriptions_returns_dict(registry: FormatRegistry) -> None:
    """Test that get_format_descriptions returns a dictionary.

    Learning notes:
    - Testing dictionary structure
    - Verifying nested data structures
    """
    descriptions = registry.get_format_descriptions()

    assert isinstance(descriptions, dict)
    assert len(descriptions) > 0


def test_format_descriptions_structure(registry: FormatRegistry) -> None:
    """Test the structure of format description entries.

    Learning notes:
    - Testing nested dictionary structure
    - Verifying all required keys are present
    - Type checking nested values
    """
    descriptions = registry.get_format_descriptions()

    # Get one format's description for detailed checking
    oxford_desc = descriptions.get("oxford")
    assert oxford_desc is not None

    # Check all required keys are present
    required_keys = [
        "display_name",
        "description",
        "side_a_label",
        "side_b_label",
        "side_a_description",
        "side_b_description",
    ]

    for key in required_keys:
        assert key in oxford_desc, f"Missing required key: {key}"
        assert isinstance(oxford_desc[key], str), f"{key} should be a string"
        assert len(oxford_desc[key]) > 0, f"{key} should not be empty"


def test_all_formats_in_descriptions(registry: FormatRegistry) -> None:
    """Test that all registered formats have descriptions.

    Learning notes:
    - Comparing two lists/sets for equality
    - Using set operations for comparison
    """
    formats = registry.list_formats()
    descriptions = registry.get_format_descriptions()

    # All formats should have descriptions
    assert set(formats) == set(descriptions.keys())


# =============================================================================
# CUSTOM FORMAT REGISTRATION TESTS
# =============================================================================


class MockDebateFormat(DebateFormat):
    """A mock format for testing custom registration.

    Learning notes:
    - Creating mock/stub objects for testing
    - Implementing minimal interface for test purposes
    - In strict mode, all abstract methods need proper return type annotations
    """

    @property
    def name(self) -> str:
        return "mock_format"

    @property
    def display_name(self) -> str:
        return "Mock Format"

    @property
    def description(self) -> str:
        return "A mock format for testing"

    def get_phases(self, participants: list[str]) -> list[FormatPhase]:
        return []

    def get_position_assignments(self, participants: list[str]) -> dict[str, Position]:
        return {}

    def get_format_instructions(self) -> str:
        return "Mock instructions"

    def get_side_labels(self, participants: list[str]) -> dict[str, str]:
        return {}

    def get_side_descriptions(self, participants: list[str]) -> dict[str, str]:
        return {}


def test_register_custom_format(registry: FormatRegistry) -> None:
    """Test that custom formats can be registered.

    Learning notes:
    - Testing extension points/plugin systems
    - Verifying state changes after operations
    """
    # Register custom format
    registry.register(MockDebateFormat)

    # Should now be in the list
    formats = registry.list_formats()
    assert "mock_format" in formats

    # Should be retrievable
    mock_format = registry.get_format("mock_format")
    assert isinstance(mock_format, MockDebateFormat)
    assert mock_format.name == "mock_format"


def test_custom_format_does_not_affect_other_registries() -> None:
    """Test that registering in one registry doesn't affect others.

    Learning notes:
    - Testing isolation between instances
    - Verifying that modifications don't leak
    - Not using the fixture here - creating instances directly
    """
    registry1 = FormatRegistry()
    registry2 = FormatRegistry()

    # Register in first registry only
    registry1.register(MockDebateFormat)

    # Should be in first registry
    assert "mock_format" in registry1.list_formats()

    # Should NOT be in second registry
    assert "mock_format" not in registry2.list_formats()


# =============================================================================
# ORGANIZED TEST CLASS - Grouping related tests
# =============================================================================


class TestFormatRetrieval:
    """Tests focused on format retrieval behavior.

    Learning notes:
    - Organizing related tests in a class
    - Classes don't need to inherit from anything
    - Fixtures work the same way with methods
    - Class name should start with 'Test'
    - Method names should start with 'test_'
    - Methods in test classes also need type annotations in strict mode
    """

    def test_oxford_format_retrieval(self, registry: FormatRegistry) -> None:
        """Test retrieving Oxford format specifically."""
        oxford = registry.get_format("oxford")
        assert oxford.name == "oxford"
        assert "Oxford" in oxford.display_name

    def test_parliamentary_format_retrieval(self, registry: FormatRegistry) -> None:
        """Test retrieving Parliamentary format specifically."""
        parliamentary = registry.get_format("parliamentary")
        assert parliamentary.name == "parliamentary"
        assert "Parliamentary" in parliamentary.display_name

    def test_case_sensitive_format_names(self, registry: FormatRegistry) -> None:
        """Test that format names are case-sensitive.

        Learning notes:
        - Testing edge cases
        - Verifying that uppercase doesn't work
        """
        # Lowercase should work
        registry.get_format("oxford")

        # Uppercase should fail
        with pytest.raises(ValueError):
            registry.get_format("OXFORD")

        with pytest.raises(ValueError):
            registry.get_format("Oxford")


class TestFormatProperties:
    """Tests for format property validation.

    Learning notes:
    - Another example of test organization
    - Testing properties/attributes of returned objects
    """

    @pytest.mark.parametrize(
        "format_name", ["oxford", "parliamentary", "socratic", "public_forum"]
    )
    def test_formats_have_valid_names(
        self, registry: FormatRegistry, format_name: str
    ) -> None:
        """Test that format name property matches registry key."""
        debate_format = registry.get_format(format_name)
        assert debate_format.name == format_name

    @pytest.mark.parametrize(
        "format_name", ["oxford", "parliamentary", "socratic", "public_forum"]
    )
    def test_formats_have_format_instructions(
        self, registry: FormatRegistry, format_name: str
    ) -> None:
        """Test that all formats provide instructions."""
        debate_format = registry.get_format(format_name)
        instructions = debate_format.get_format_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 0


# =============================================================================
# EDGE CASES AND BOUNDARY TESTS
# =============================================================================


def test_list_formats_returns_copy_not_reference(registry: FormatRegistry) -> None:
    """Test that list_formats returns a new list each time.

    Learning notes:
    - Testing for defensive copying
    - Ensuring internal state isn't exposed
    """
    formats1 = registry.list_formats()
    formats2 = registry.list_formats()

    # Should be equal but not the same object
    assert formats1 == formats2
    assert formats1 is not formats2


def test_format_descriptions_with_empty_participants_list(
    registry: FormatRegistry,
) -> None:
    """Test that format descriptions work even with edge case inputs.

    Learning notes:
    - Testing edge cases
    - The method uses dummy participants internally
    """
    # Should not raise an exception
    descriptions = registry.get_format_descriptions()
    assert len(descriptions) > 0


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


def test_full_format_workflow(
    registry: FormatRegistry, sample_participants: list[str]
) -> None:
    """Test a complete workflow of listing, getting, and using a format.

    Learning notes:
    - Testing workflows/scenarios end-to-end
    - Combining multiple operations
    - More realistic usage testing
    """
    # List available formats
    formats = registry.list_formats()
    assert "oxford" in formats

    # Get a specific format
    oxford = registry.get_format("oxford")
    assert oxford.name == "oxford"

    # Use the format
    phases = oxford.get_phases(sample_participants)
    assert len(phases) > 0

    side_labels = oxford.get_side_labels(sample_participants)
    assert len(side_labels) == 2

    instructions = oxford.get_format_instructions()
    assert len(instructions) > 0


def test_all_formats_complete_workflow(
    registry: FormatRegistry, sample_participants: list[str]
) -> None:
    """Test that all formats can be retrieved and used.

    Learning notes:
    - Iterating over dynamic data
    - Testing all items in a collection
    - Comprehensive validation
    """
    for format_name in registry.list_formats():
        # Should be retrievable
        debate_format = registry.get_format(format_name)
        assert debate_format.name == format_name

        # Should provide all required information
        assert debate_format.display_name
        assert debate_format.description
        assert debate_format.get_format_instructions()

        # Should work with participants
        phases = debate_format.get_phases(sample_participants)
        assert isinstance(phases, list)

        side_labels = debate_format.get_side_labels(sample_participants)
        assert isinstance(side_labels, dict)


# =============================================================================
# NOTES ON RUNNING TESTS
# =============================================================================

"""
To run these tests:

1. Run all tests in this file:
   pytest tests/test_format_registry.py

2. Run with verbose output:
   pytest tests/test_format_registry.py -v

3. Run a specific test:
   pytest tests/test_format_registry.py::test_registry_initialization

4. Run tests in a class:
   pytest tests/test_format_registry.py::TestFormatRetrieval

5. Run with coverage:
   pytest tests/test_format_registry.py --cov=dialectus.engine.formats.registry

6. Run and stop at first failure:
   pytest tests/test_format_registry.py -x

7. Run and show print statements:
   pytest tests/test_format_registry.py -s

8. Run tests matching a pattern:
   pytest tests/test_format_registry.py -k "oxford"
"""
