from models.base_types import BaseModel


class OpenRouterArchitecture(BaseModel):
    input_modalities: list[str]
    output_modalities: list[str]
    tokenizer: str
    instruct_type: str | None = None
