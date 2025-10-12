from dialectus.engine.models.base_types import BaseModel


class OpenRouterPricing(BaseModel):
    prompt: str
    completion: str
    image: str = "0"
    request: str = "0"
    web_search: str = "0"
    internal_reasoning: str = "0"
    input_cache_read: str = "0"
    input_cache_write: str = "0"

    @property
    def prompt_cost_per_1k(self) -> float:
        """Cost per 1K prompt tokens in dollars."""
        return float(self.prompt) * 1000 if self.prompt != "0" else 0.0

    @property
    def completion_cost_per_1k(self) -> float:
        """Cost per 1K completion tokens in dollars."""
        return float(self.completion) * 1000 if self.completion != "0" else 0.0

    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1K tokens (assuming 50/50 prompt/completion)."""
        return (self.prompt_cost_per_1k + self.completion_cost_per_1k) / 2

    @property
    def is_free(self) -> bool:
        """Check if model is completely free."""
        return float(self.prompt) == 0.0 and float(self.completion) == 0.0
