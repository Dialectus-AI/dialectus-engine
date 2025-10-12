"""Configuration settings and data models."""

from typing import Literal
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class ModelConfig(BaseModel):
    """Configuration for a debate model."""

    name: str = Field(
        ...,
        description="Model name (e.g., 'llama3.2:3b' for Ollama, 'openai/gpt-4' for OpenRouter)",
    )
    provider: str = Field(
        default="ollama", description="Model provider (ollama, openrouter, etc.)"
    )
    personality: str = Field(default="neutral", description="Debate personality style")
    max_tokens: int = Field(default=300, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, description="Model temperature")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str):
        valid_providers = {"ollama", "openrouter"}
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of: {valid_providers}")
        return v


class DebateConfig(BaseModel):
    """Main debate configuration."""

    topic: str = Field(..., description="Debate topic")
    format: str = Field(default="oxford", description="Debate format")
    time_per_turn: int = Field(default=120, description="Seconds per turn")
    word_limit: int = Field(default=200, description="Word limit per turn")


class JudgingConfig(BaseModel):
    """Judging system configuration."""

    criteria: list[str] = Field(
        default=["logic", "evidence", "persuasiveness"], description="Scoring criteria"
    )
    judge_models: list[str] = Field(
        default=[], description="Models to use for AI judging"
    )
    judge_provider: str | None = Field(
        default=None, description="Provider for the judge models"
    )


class OllamaConfig(BaseModel):
    """Ollama-specific configuration for hardware optimization."""

    num_gpu_layers: int | None = Field(
        default=None,
        description="Number of layers to offload to GPU (-1 for all, 0 for CPU-only)",
    )
    gpu_memory_utilization: float | None = Field(
        default=None, description="GPU memory utilization percentage (0.0-1.0)"
    )
    main_gpu: int | None = Field(
        default=None, description="Primary GPU device ID for multi-GPU setups"
    )
    num_thread: int | None = Field(
        default=None, description="Number of CPU threads for processing"
    )
    keep_alive: str | None = Field(
        default="5m",
        description="How long to keep models loaded (e.g., '5m', '1h', '0' for immediate unload)",
    )
    repeat_penalty: float | None = Field(
        default=1.1, description="Penalty for repetition in responses"
    )


class OpenRouterConfig(BaseModel):
    """OpenRouter-specific configuration."""

    api_key: str | None = Field(
        default=None,
        description="OpenRouter API key (can also be set via OPENROUTER_API_KEY env var)",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="OpenRouter API base URL"
    )
    site_url: str | None = Field(
        default=None, description="Your site URL for OpenRouter referrer tracking"
    )
    app_name: str | None = Field(
        default="Dialectus AI Debate System",
        description="App name for OpenRouter tracking",
    )
    max_retries: int = Field(
        default=3, description="Maximum number of API call retries"
    )
    timeout: int = Field(default=60, description="API request timeout in seconds")


class SystemConfig(BaseModel):
    """System-wide configuration."""

    # Ollama configuration (backward compatibility)
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API URL"
    )
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig, description="Ollama-specific settings"
    )

    # Provider configurations
    openrouter: OpenRouterConfig = Field(
        default_factory=OpenRouterConfig, description="OpenRouter-specific settings"
    )

    # System settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Topic generation settings
    debate_topic_source: Literal["ollama", "openrouter"] = Field(
        default="openrouter", description="Provider for topic generation model"
    )
    debate_topic_model: str = Field(
        default="anthropic/claude-3-haiku",
        description="Model name for topic generation",
    )

    @field_validator("debate_topic_source")
    @classmethod
    def validate_topic_source(cls, v: str):
        valid_sources = {"ollama", "openrouter"}
        if v not in valid_sources:
            raise ValueError(f"debate_topic_source must be one of: {valid_sources}")
        return v


class AppConfig(BaseModel):
    """Complete application configuration."""

    debate: DebateConfig
    models: dict[str, ModelConfig]
    judging: JudgingConfig
    system: SystemConfig

    @classmethod
    def load_from_file(cls, config_path: Path) -> "AppConfig":
        """Load configuration from JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        import json

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required sections
        required_sections = ["debate", "models", "judging", "system"]
        missing_sections = [
            section for section in required_sections if section not in data
        ]
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")

        # Validate models section has at least one model
        if not data.get("models") or len(data["models"]) == 0:
            raise ValueError(
                "Config must include at least one model in 'models' section"
            )

        return cls(**data)


def get_default_config() -> AppConfig:
    """Load default configuration from debate_config.json, creating it if needed."""
    config_path = Path("debate_config.json")
    if not config_path.exists():
        # Auto-create from debate_config.example.json if it exists
        example_path = Path("debate_config.example.json")
        if example_path.exists():
            import shutil

            shutil.copy2(example_path, config_path)
        else:
            # Fallback to template config
            template_config = get_template_config()
            import json

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(template_config.model_dump(exclude_unset=True), f, indent=2)
    return AppConfig.load_from_file(config_path)


def get_template_config() -> AppConfig:
    """Get template configuration for config file generation."""
    return AppConfig(
        debate=DebateConfig(
            topic="Should artificial intelligence be regulated by government oversight?",
            format="oxford",
            time_per_turn=120,
            word_limit=100,
        ),
        models={
            "model_a": ModelConfig(
                name="qwen2.5:7b",
                provider="ollama",
                personality="analytical",
                max_tokens=300,
                temperature=0.7,
            ),
            "model_b": ModelConfig(
                name="openai/gpt-4",
                provider="openrouter",
                personality="passionate",
                max_tokens=300,
                temperature=0.8,
            ),
        },
        judging=JudgingConfig(
            criteria=["logic", "evidence", "persuasiveness"],
            judge_models=["openthinker:7b"],
            judge_provider="ollama",
        ),
        system=SystemConfig(
            ollama_base_url="http://localhost:11434",
            ollama=OllamaConfig(
                num_gpu_layers=-1,  # Use all GPU layers by default
                gpu_memory_utilization=None,
                main_gpu=None,
                num_thread=None,
                keep_alive="5m",
                repeat_penalty=1.1,
            ),
            openrouter=OpenRouterConfig(
                api_key=None,  # Set your OpenRouter API key here or use OPENROUTER_API_KEY env var
                base_url="https://openrouter.ai/api/v1",
                site_url=None,  # Your site URL for referrer tracking (optional)
                app_name="Dialectus AI Debate System",
                max_retries=3,
                timeout=60,
            ),
            log_level="INFO",
            debate_topic_source="openrouter",
            debate_topic_model="anthropic/claude-3-haiku",
        ),
    )
