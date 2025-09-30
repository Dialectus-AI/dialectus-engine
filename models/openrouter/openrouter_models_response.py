from models.base_types import BaseModel
from models.openrouter.openrouter_model import OpenRouterModel


class OpenRouterModelsResponse(BaseModel):
    data: list[OpenRouterModel]