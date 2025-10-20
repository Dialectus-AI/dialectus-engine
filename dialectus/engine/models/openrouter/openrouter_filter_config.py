from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict, cast


class _FilterPatternDict(TypedDict, total=False):
    """Typed representation of each filter category stored in JSON."""

    category: str
    patterns: list[str]


class _FilterConfigDict(TypedDict, total=False):
    """Typed representation of the filters subsection in the JSON config."""

    exclude_patterns: list[_FilterPatternDict]
    preview_patterns: list[_FilterPatternDict]


class _SettingsDict(TypedDict, total=False):
    """Typed representation of configurable settings."""

    allow_preview_models: bool
    exclude_free_tier_models: bool
    max_cost_per_1k_tokens: float
    min_context_length: int
    max_models_per_tier: int


class _ConfigDict(TypedDict, total=False):
    """Top-level configuration schema stored on disk."""

    filters: _FilterConfigDict
    settings: _SettingsDict


logger = logging.getLogger(__name__)


class OpenRouterFilterConfig:
    """Loads and manages OpenRouter filtering configuration from external JSON file."""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            # Default to config/openrouter_filters.json relative to this file
            config_path = (
                Path(__file__).parent.parent.parent
                / "config"
                / "openrouter_filters.json"
            )

        self.config_path = config_path
        self._config: _ConfigDict | None = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file with fallback to defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    loaded_config = json.load(f)
                if isinstance(loaded_config, dict):
                    self._config = cast(_ConfigDict, loaded_config)
                    logger.info(
                        "Loaded OpenRouter filter config from %s", self.config_path
                    )
                else:
                    logger.warning(
                        "OpenRouter filter config at %s has unexpected format; using"
                        " defaults",
                        self.config_path,
                    )
                    self._config = self._get_default_config()
            else:
                logger.warning(
                    "Filter config file not found at %s, using defaults",
                    self.config_path,
                )
                self._config = self._get_default_config()
        except Exception as error:
            logger.error(
                "Failed to load filter config from %s: %s",
                self.config_path,
                error,
            )
            self._config = self._get_default_config()

    def _get_default_config(self) -> _ConfigDict:
        """Fallback default configuration if JSON file is missing/invalid."""
        return {
            "filters": {
                "exclude_patterns": [
                    {
                        "category": "meta_routing_models",
                        "patterns": [
                            "auto.*router",
                            "router",
                            "meta.*llama.*auto",
                            "mixture.*expert",
                            "moe",
                        ],
                    }
                ],
                "preview_patterns": [
                    {
                        "category": "preview_anonymous_models",
                        "patterns": [
                            "^preview-",
                            "^anonymous-",
                            "^beta-",
                            "-preview$",
                            "-beta$",
                        ],
                    }
                ],
            },
            "settings": {
                "allow_preview_models": False,
                "exclude_free_tier_models": False,
                "max_cost_per_1k_tokens": 0.02,
                "min_context_length": 4096,
                "max_models_per_tier": 8,
            },
        }

    def get_exclude_patterns(self) -> list[str]:
        """Get all exclusion patterns as a flat list."""
        patterns: list[str] = []
        if not self._config:
            return patterns

        filters = self._config.get("filters")
        if not filters:
            return patterns

        # JSON parsing always yields `dict[str, object]`; we re-cast to keep
        # pyright aware of concrete key/value shapes and avoid Unknown bleed-through.
        raw_categories = filters.get("exclude_patterns")
        if not isinstance(raw_categories, list):
            return patterns

        for category_dict in cast(list[dict[str, object]], raw_categories):
            # `typing.cast` keeps these lists well-typed without mutating
            # runtime behaviour.
            raw_patterns_obj = category_dict.get("patterns")
            if not isinstance(raw_patterns_obj, list):
                continue

            raw_patterns = cast(list[object], raw_patterns_obj)

            for pattern_obj in raw_patterns:
                if isinstance(pattern_obj, str):
                    patterns.append(pattern_obj)

        return patterns

    def get_preview_patterns(self) -> list[str]:
        """Get all preview detection patterns as a flat list."""
        patterns: list[str] = []
        if not self._config:
            return patterns

        filters = self._config.get("filters")
        if not filters:
            return patterns

        # Same defensive typing story as above for preview-pattern categories.
        raw_categories = filters.get("preview_patterns")
        if not isinstance(raw_categories, list):
            return patterns

        for category_dict in cast(list[dict[str, object]], raw_categories):
            raw_patterns_obj = category_dict.get("patterns")
            if not isinstance(raw_patterns_obj, list):
                continue

            raw_patterns = cast(list[object], raw_patterns_obj)

            for pattern_obj in raw_patterns:
                if isinstance(pattern_obj, str):
                    patterns.append(pattern_obj)

        return patterns

    def get_setting(
        self,
        setting_name: str,
        default: bool | int | float | str | None = None,
    ) -> bool | int | float | str | None:
        """Get a setting value with fallback to default.

        Priority:
        1. Environment variable (OPENROUTER_<SETTING_NAME> in uppercase)
        2. Config file setting
        3. Default value
        """
        import os

        # Check environment variable first
        env_var_name = f"OPENROUTER_{setting_name.upper()}"
        env_value = os.getenv(env_var_name)

        if env_value is not None:
            # Convert string env var to appropriate type
            if env_value.lower() in ("true", "1", "yes"):
                return True
            elif env_value.lower() in ("false", "0", "no"):
                return False
            elif env_value.replace(".", "", 1).isdigit():
                # Numeric value
                return float(env_value) if "." in env_value else int(env_value)
            return env_value

        # Fall back to config file
        if not self._config:
            return default

        settings = self._config.get("settings")
        if not settings:
            return default

        value: object = settings.get(setting_name, default)
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value

        logger.warning(
            "Setting %s in %s has unsupported type %s; using default",
            setting_name,
            self.config_path,
            type(value).__name__,
        )
        return default

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
