from dialectus.engine.models.base_types import BaseModel

class OpenRouterTopProvider(BaseModel):
    is_moderated: bool
    context_length: int | None = None
    max_completion_tokens: int | None = None

