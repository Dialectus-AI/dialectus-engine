from pathlib import Path
import logging
import json
from typing import Any

logger = logging.getLogger(__name__)

class OpenRouterFilterConfig:
    """Loads and manages OpenRouter filtering configuration from external JSON file."""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            # Default to config/openrouter_filters.json relative to this file
            config_path = (
                Path(__file__).parent.parent / "config" / "openrouter_filters.json"
            )

        self.config_path = config_path
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file with fallback to defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info(f"Loaded OpenRouter filter config from {self.config_path}")
            else:
                logger.warning(
                    f"Filter config file not found at {self.config_path}, using defaults"
                )
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load filter config from {self.config_path}: {e}")
            self._config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
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
        patterns = []
        filters = self._config.get("filters", {}) if self._config else {}
        for category in filters.get("exclude_patterns", []):
            patterns.extend(category.get("patterns", []))
        return patterns

    def get_preview_patterns(self) -> list[str]:
        """Get all preview detection patterns as a flat list."""
        patterns = []
        filters = self._config.get("filters", {}) if self._config else {}
        for category in filters.get("preview_patterns", []):
            patterns.extend(category.get("patterns", []))
        return patterns

    def get_setting(self, setting_name: str, default=None):
        """Get a setting value with fallback to default.

        Priority:
        1. Environment variable (OPENROUTER_<SETTING_NAME> in uppercase)
        2. Config file setting
        3. Default value
        """
        import os

        # Check environment variable first (Railway/production)
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
        if self._config is None:
            return default
        return self._config.get("settings", {}).get(setting_name, default)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

