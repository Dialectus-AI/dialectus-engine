"""Registry for debate formats."""

from .base import DebateFormat
from .oxford import OxfordFormat
from .parliamentary import ParliamentaryFormat
from .socratic import SocraticFormat
from .public_forum import PublicForumFormat


class FormatRegistry:
    """Registry for managing available debate formats."""

    def __init__(self):
        self._formats: dict[str, type[DebateFormat]] = {}
        self._register_built_in_formats()

    def _register_built_in_formats(self):
        """Register the built-in debate formats."""
        self.register(OxfordFormat)
        self.register(ParliamentaryFormat)
        self.register(SocraticFormat)
        self.register(PublicForumFormat)

    def register(self, format_class: type[DebateFormat]) -> None:
        """Register a debate format class."""
        instance = format_class()
        self._formats[instance.name] = format_class

    def get_format(self, name: str) -> DebateFormat:
        """Get a format instance by name."""
        if name not in self._formats:
            raise ValueError(
                f"Unknown format: {name}. Available: {list(self._formats.keys())}"
            )
        return self._formats[name]()

    def list_formats(self) -> list[str]:
        """List all available format names."""
        return list(self._formats.keys())

    def get_format_descriptions(self) -> dict[str, dict[str, str]]:
        """Get format names, display names, descriptions, and side labels with descriptions."""
        descriptions: dict[str, dict[str, str]] = {}
        for name, format_class in self._formats.items():
            instance = format_class()
            # Use dummy participants to get side labels and descriptions
            dummy_participants = ["participant_a", "participant_b"]
            side_labels = instance.get_side_labels(dummy_participants)
            side_descriptions = instance.get_side_descriptions(dummy_participants)

            descriptions[name] = {
                "display_name": instance.display_name,
                "description": instance.description,
                "side_a_label": side_labels.get("participant_a", "Pro"),
                "side_b_label": side_labels.get("participant_b", "Con"),
                "side_a_description": side_descriptions.get("participant_a", "This AI will argue in favor of the topic."),
                "side_b_description": side_descriptions.get("participant_b", "This AI will argue against the topic."),
            }
        return descriptions


# Global registry instance
format_registry = FormatRegistry()
