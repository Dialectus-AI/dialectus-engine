from dialectus.engine.models.base_types import BaseModel
from dialectus.engine.models.openrouter.openrouter_model import OpenRouterModel


class OpenRouterModelsResponse(BaseModel):
    data: list[OpenRouterModel]