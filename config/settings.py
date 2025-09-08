"""Configuration settings and data models."""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field
import yaml
from pathlib import Path


class ModelConfig(BaseModel):
    """Configuration for a debate model."""

    name: str = Field(..., description="Ollama model name (e.g., 'llama3.2:3b')")
    personality: str = Field(default="neutral", description="Debate personality style")
    max_tokens: int = Field(default=300, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, description="Model temperature")


class DebateConfig(BaseModel):
    """Main debate configuration."""

    topic: str = Field(..., description="Debate topic")
    format: Literal["parliamentary", "oxford", "socratic", "custom"] = Field(
        default="oxford", description="Debate format"
    )
    time_per_turn: int = Field(default=120, description="Seconds per turn")
    word_limit: int = Field(default=200, description="Word limit per turn")


class JudgingConfig(BaseModel):
    """Judging system configuration."""

    method: Literal["ai", "ensemble", "none"] = Field(
        default="ai", description="Judging method"
    )
    criteria: List[str] = Field(
        default=["logic", "evidence", "persuasiveness"], description="Scoring criteria"
    )
    judge_model: Optional[str] = Field(
        default=None, description="Model to use for AI judging"
    )


class OllamaConfig(BaseModel):
    """Ollama-specific configuration for hardware optimization."""
    
    num_gpu_layers: Optional[int] = Field(
        default=None, description="Number of layers to offload to GPU (-1 for all, 0 for CPU-only)"
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=None, description="GPU memory utilization percentage (0.0-1.0)"
    )
    main_gpu: Optional[int] = Field(
        default=None, description="Primary GPU device ID for multi-GPU setups"
    )
    num_thread: Optional[int] = Field(
        default=None, description="Number of CPU threads for processing"
    )
    keep_alive: Optional[str] = Field(
        default="5m", description="How long to keep models loaded (e.g., '5m', '1h', '0' for immediate unload)"
    )
    repeat_penalty: Optional[float] = Field(
        default=1.1, description="Penalty for repetition in responses"
    )


class SystemConfig(BaseModel):
    """System-wide configuration."""

    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API URL"
    )
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig, description="Ollama-specific settings"
    )
    save_transcripts: bool = Field(
        default=True, description="Save debate transcripts to disk"
    )
    transcript_dir: str = Field(
        default="transcripts", description="Directory to save transcripts"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


class AppConfig(BaseModel):
    """Complete application configuration."""

    debate: DebateConfig
    models: Dict[str, ModelConfig]
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

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(exclude_unset=True),
                f,
                default_flow_style=False,
                indent=2,
            )


def get_default_config() -> AppConfig:
    """Load default configuration from debate_config.json, creating it if needed."""
    config_path = Path("debate_config.json")
    if not config_path.exists():
        # Auto-create from example.config.json if it exists
        example_path = Path("example.config.json")
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
            word_limit=200,
        ),
        models={
            "model_a": ModelConfig(
                name="qwen2.5:7b",
                personality="analytical",
                max_tokens=300,
                temperature=0.7,
            ),
            "model_b": ModelConfig(
                name="qwen2.5:7b",
                personality="passionate",
                max_tokens=300,
                temperature=0.8,
            ),
        },
        judging=JudgingConfig(
            method="ai",
            criteria=["logic", "evidence", "persuasiveness"],
            judge_model="openthinker:7b",
        ),
        system=SystemConfig(
            ollama_base_url="http://localhost:11434",
            ollama=OllamaConfig(
                num_gpu_layers=-1,  # Use all GPU layers by default
                keep_alive="5m"
            ),
            log_level="INFO",
        ),
    )
