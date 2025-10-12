
from dialectus.engine.models.base_types import ModelWeightClass
from dialectus.engine.models.openrouter.openrouter_model import OpenRouterModel


class OpenRouterCapabilityExtractor:
    """Extract model capabilities from OpenRouter API data automatically."""

    @classmethod
    def extract_parameter_count(cls, model: OpenRouterModel) -> str | None:
        """Extract parameter count from model name, description, or hugging_face_id."""
        import re

        # Common sources to check
        sources = [model.name, model.id, model.description]
        if model.hugging_face_id:
            sources.append(model.hugging_face_id)

        # Parameter patterns to match (in order of preference)
        param_patterns = [
            r"(\d+(?:\.\d+)?)B[^a-zA-Z]",  # "7B ", "70.5B-"
            r"(\d+(?:\.\d+)?)b[^a-zA-Z]",  # "7b ", "13b-"
            r"(\d+(?:\.\d+)?)B$",  # "405B" at end
            r"(\d+(?:\.\d+)?)b$",  # "7b" at end
            r"(\d+(?:\.\d+)?)T[^a-zA-Z]",  # "1T " (trillion parameters)
            r"(\d+(?:\.\d+)?)M[^a-zA-Z]",  # "500M " (million parameters)
        ]

        for source in sources:
            source_lower = source.lower()

            for pattern in param_patterns:
                match = re.search(pattern, source_lower)
                if match:
                    size = float(match.group(1))

                    # Determine unit from pattern
                    if "T" in pattern.upper():
                        return f"{size:.1f}T" if size != int(size) else f"{int(size)}T"
                    elif "M" in pattern.upper():
                        # Convert millions to billions if over 1000M
                        if size >= 1000:
                            return f"{size/1000:.1f}B"
                        return f"{size:.0f}M"
                    else:  # B (billions)
                        return f"{size:.1f}B" if size != int(size) else f"{int(size)}B"

        # Special case patterns for known model families
        name_lower = model.name.lower()
        id_lower = model.id.lower()

        # Well-known model sizes
        known_sizes = {
            "tiny": "1B",
            "small": "7B",
            "medium": "13B",
            "large": "70B",
            "xl": "405B",
            "nano": "1B",
            "micro": "3B",
            "mini": "7B",
            "base": "7B",
        }

        for size_name, param_count in known_sizes.items():
            if size_name in name_lower or size_name in id_lower:
                return param_count

        return None

    @classmethod
    def determine_weight_class(
        cls, model: OpenRouterModel, estimated_params: str | None = None
    ) -> ModelWeightClass:
        """Determine weight class based on context length, parameters, and capabilities."""

        # Get parameter count if not provided
        if not estimated_params:
            estimated_params = cls.extract_parameter_count(model)

        context_length = model.context_length

        # Parameter-based classification (takes precedence)
        if estimated_params:
            param_value = float(estimated_params.rstrip("BMT"))
            param_unit = estimated_params[-1].upper()

            # Convert to billions for comparison
            if param_unit == "T":
                param_billions = param_value * 1000
            elif param_unit == "M":
                param_billions = param_value / 1000
            else:  # B
                param_billions = param_value

            # Parameter-based thresholds for debate models
            if param_billions >= 100:  # 100B+ parameters
                return ModelWeightClass.ULTRAWEIGHT
            elif param_billions >= 30:  # 30B-100B
                return ModelWeightClass.HEAVYWEIGHT
            elif param_billions >= 5:  # 5B-30B
                return ModelWeightClass.MIDDLEWEIGHT
            else:  # <5B
                return ModelWeightClass.LIGHTWEIGHT

        # Context-based fallback classification
        if context_length >= 128000:  # 128K+ context
            return ModelWeightClass.ULTRAWEIGHT
        elif context_length >= 32000:  # 32K-128K
            return ModelWeightClass.HEAVYWEIGHT
        elif context_length >= 8000:  # 8K-32K
            return ModelWeightClass.MIDDLEWEIGHT
        else:  # <8K context
            return ModelWeightClass.LIGHTWEIGHT

    @classmethod
    def calculate_debate_score(
        cls, model: OpenRouterModel, weight_class: ModelWeightClass
    ) -> float:
        """Calculate debate suitability score based on conversational and reasoning capabilities."""

        # Base score from context length (important for following debate flow and context)
        context_score = min(
            model.context_length / 32000, 3.0
        )  # Cap at 3x for 32K+ context

        # Cost efficiency (lower cost = higher score for debates)
        if model.pricing.is_free:
            cost_score = 5.0  # High score for free models
        else:
            avg_cost = max(model.pricing.avg_cost_per_1k, 0.0001)
            cost_score = min(0.005 / avg_cost, 5.0)  # Normalize around $0.005/1K

        # Weight class bonus (larger models generally better at reasoning and debate)
        weight_bonus = {
            ModelWeightClass.ULTRAWEIGHT: 1.5,
            ModelWeightClass.HEAVYWEIGHT: 1.3,
            ModelWeightClass.MIDDLEWEIGHT: 1.0,
            ModelWeightClass.LIGHTWEIGHT: 0.7,  # Lower score for small models in debate context
        }.get(weight_class, 1.0)

        # Conversational suitability bonus
        conversational_bonus = cls._calculate_conversational_bonus(model)

        # Multimodal input capability bonus (often indicates more sophisticated models)
        multimodal_bonus = 1.0
        if (
            "text" in model.architecture.input_modalities
            and "text" in model.architecture.output_modalities
        ):
            # Count non-text input modalities
            non_text_inputs = [
                m for m in model.architecture.input_modalities if m != "text"
            ]
            non_text_outputs = [
                m for m in model.architecture.output_modalities if m != "text"
            ]

            if len(non_text_outputs) == 0:  # Text-only output (required for debate)
                if len(non_text_inputs) > 0:  # Multimodal input, text output (ideal)
                    multimodal_bonus = 1.15  # Bonus for advanced multimodal models like GPT-4V, Claude-3
                else:  # Text-only input and output
                    multimodal_bonus = 1.0  # Standard text models
            # Note: Models with non-text outputs filtered out earlier

        return (
            context_score
            * cost_score
            * weight_bonus
            * conversational_bonus
            * multimodal_bonus
        )

    @classmethod
    def _calculate_conversational_bonus(cls, model: OpenRouterModel) -> float:
        """Calculate bonus score based on model's conversational capabilities."""
        model_name_lower = model.name.lower()
        description_lower = model.description.lower()

        # Positive indicators for debate/conversation
        positive_keywords = [
            "chat",
            "instruct",
            "assistant",
            "conversation",
            "dialogue",
            "gpt",
            "claude",
            "gemini",
            "llama",
            "mistral",
            "qwen",
            "roleplay",
            "character",
            "persona",
            "uncensored",  # Good for taking debate positions
        ]

        # Negative indicators for debate (specialized models)
        negative_keywords = [
            "code",
            "math",
            "scientific",
            "translation",
            "embed",
            "vision",
            "image",
            "audio",
            "video",
            "jailbreak",
        ]

        bonus = 1.0

        # Boost for conversational indicators
        for keyword in positive_keywords:
            if keyword in model_name_lower or keyword in description_lower:
                bonus += 0.1

        # Penalty for specialized/non-conversational indicators
        for keyword in negative_keywords:
            if keyword in model_name_lower or keyword in description_lower:
                bonus -= 0.2

        # Bonus for models with "chat" or "instruct" in name (strong conversational indicators)
        if any(word in model_name_lower for word in ["chat", "instruct"]):
            bonus += 0.3

        # Bonus for roleplay models (excellent for taking debate positions/personas)
        if any(
            word in model_name_lower for word in ["roleplay", "character", "persona"]
        ):
            bonus += 0.2

        # Bonus for uncensored models (may be more willing to take strong positions)
        if "uncensored" in model_name_lower:
            bonus += 0.15

        # Penalty for base/raw models (usually not fine-tuned for conversation)
        if "base" in model_name_lower and "instruct" not in model_name_lower:
            bonus -= 0.3

        # Ensure bonus stays in reasonable range
        return max(0.5, min(bonus, 2.0))

