"""
OpenRouter API types and model filtering logic.
Handles model selection, pricing analysis, and weight classification.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import re
from decimal import Decimal
from .base_types import ModelWeightClass, ModelTier, ModelPricing, BaseEnhancedModelInfo


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


class OpenRouterArchitecture(BaseModel):
    input_modalities: List[str]
    output_modalities: List[str]
    tokenizer: str
    instruct_type: Optional[str] = None


class OpenRouterTopProvider(BaseModel):
    is_moderated: bool
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None


class OpenRouterModel(BaseModel):
    id: str
    name: str
    created: int
    description: str
    architecture: OpenRouterArchitecture
    top_provider: OpenRouterTopProvider
    pricing: OpenRouterPricing
    canonical_slug: str
    context_length: int
    hugging_face_id: Optional[str] = None
    per_request_limits: Optional[Dict[str, Any]] = None
    supported_parameters: List[str] = Field(default_factory=list)


class OpenRouterModelsResponse(BaseModel):
    data: List[OpenRouterModel]


class OpenRouterEnhancedModelInfo(BaseEnhancedModelInfo):
    """Enhanced OpenRouter model information extending base class."""
    
    def __init__(self, **data):
        # Convert OpenRouter pricing to generic pricing
        if 'pricing' in data and isinstance(data['pricing'], OpenRouterPricing):
            openrouter_pricing = data['pricing']
            data['pricing'] = ModelPricing(
                prompt_cost_per_1k=openrouter_pricing.prompt_cost_per_1k,
                completion_cost_per_1k=openrouter_pricing.completion_cost_per_1k,
                is_free=openrouter_pricing.is_free,
                currency="USD"
            )
        
        # Store OpenRouter-specific info
        if 'openrouter_raw' not in data.get('source_info', {}):
            data.setdefault('source_info', {})['openrouter_raw'] = data.get('raw_model', {})
        
        super().__init__(**data)


class OpenRouterCapabilityExtractor:
    """Extract model capabilities from OpenRouter API data automatically."""
    
    @classmethod
    def extract_parameter_count(cls, model: OpenRouterModel) -> Optional[str]:
        """Extract parameter count from model name, description, or hugging_face_id."""
        import re
        
        # Common sources to check
        sources = [model.name, model.id, model.description]
        if model.hugging_face_id:
            sources.append(model.hugging_face_id)
        
        # Parameter patterns to match (in order of preference)
        param_patterns = [
            r'(\d+(?:\.\d+)?)B[^a-zA-Z]',  # "7B ", "70.5B-"
            r'(\d+(?:\.\d+)?)b[^a-zA-Z]',  # "7b ", "13b-"
            r'(\d+(?:\.\d+)?)B$',          # "405B" at end
            r'(\d+(?:\.\d+)?)b$',          # "7b" at end
            r'(\d+(?:\.\d+)?)T[^a-zA-Z]',  # "1T " (trillion parameters)
            r'(\d+(?:\.\d+)?)M[^a-zA-Z]',  # "500M " (million parameters)
        ]
        
        for source in sources:
            source_lower = source.lower()
            
            for pattern in param_patterns:
                match = re.search(pattern, source_lower)
                if match:
                    size = float(match.group(1))
                    
                    # Determine unit
                    unit = pattern[-4]  # Get the unit letter (B, T, M)
                    if 'T' in pattern.upper():
                        return f"{size:.1f}T" if size != int(size) else f"{int(size)}T"
                    elif 'M' in pattern.upper():
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
            'tiny': '1B', 'small': '7B', 'medium': '13B', 'large': '70B', 'xl': '405B',
            'nano': '1B', 'micro': '3B', 'mini': '7B', 'base': '7B'
        }
        
        for size_name, param_count in known_sizes.items():
            if size_name in name_lower or size_name in id_lower:
                return param_count
        
        return None
    
    @classmethod
    def determine_weight_class(cls, model: OpenRouterModel, estimated_params: Optional[str] = None) -> ModelWeightClass:
        """Determine weight class based on context length, parameters, and capabilities."""
        
        # Get parameter count if not provided
        if not estimated_params:
            estimated_params = cls.extract_parameter_count(model)
        
        context_length = model.context_length
        
        # Parameter-based classification (takes precedence)
        if estimated_params:
            param_value = float(estimated_params.rstrip('BMT'))
            param_unit = estimated_params[-1].upper()
            
            # Convert to billions for comparison
            if param_unit == 'T':
                param_billions = param_value * 1000
            elif param_unit == 'M':
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
        elif context_length >= 8000:   # 8K-32K
            return ModelWeightClass.MIDDLEWEIGHT
        else:  # <8K context
            return ModelWeightClass.LIGHTWEIGHT
    
    @classmethod
    def calculate_debate_score(cls, model: OpenRouterModel, weight_class: ModelWeightClass) -> float:
        """Calculate debate suitability score based on conversational and reasoning capabilities."""
        
        # Base score from context length (important for following debate flow and context)
        context_score = min(model.context_length / 32000, 3.0)  # Cap at 3x for 32K+ context
        
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
            ModelWeightClass.LIGHTWEIGHT: 0.7  # Lower score for small models in debate context
        }.get(weight_class, 1.0)
        
        # Conversational suitability bonus
        conversational_bonus = cls._calculate_conversational_bonus(model)
        
        # Multimodal input capability bonus (often indicates more sophisticated models)
        multimodal_bonus = 1.0
        if 'text' in model.architecture.input_modalities and 'text' in model.architecture.output_modalities:
            # Count non-text input modalities
            non_text_inputs = [m for m in model.architecture.input_modalities if m != 'text']
            non_text_outputs = [m for m in model.architecture.output_modalities if m != 'text']
            
            if len(non_text_outputs) == 0:  # Text-only output (required for debate)
                if len(non_text_inputs) > 0:  # Multimodal input, text output (ideal)
                    multimodal_bonus = 1.15  # Bonus for advanced multimodal models like GPT-4V, Claude-3
                else:  # Text-only input and output
                    multimodal_bonus = 1.0   # Standard text models
            # Note: Models with non-text outputs filtered out earlier
        
        return context_score * cost_score * weight_bonus * conversational_bonus * multimodal_bonus
    
    @classmethod
    def _calculate_conversational_bonus(cls, model: OpenRouterModel) -> float:
        """Calculate bonus score based on model's conversational capabilities."""
        model_name_lower = model.name.lower()
        model_id_lower = model.id.lower() 
        description_lower = model.description.lower()
        
        # Positive indicators for debate/conversation
        positive_keywords = [
            'chat', 'instruct', 'assistant', 'conversation', 'dialogue',
            'gpt', 'claude', 'gemini', 'llama', 'mistral', 'qwen',
            'roleplay', 'character', 'persona', 'uncensored'  # Good for taking debate positions
        ]
        
        # Negative indicators for debate (specialized models)
        negative_keywords = [
            'code', 'math', 'scientific', 'translation', 'embed',
            'vision', 'image', 'audio', 'video', 'jailbreak'
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
        if any(word in model_name_lower for word in ['chat', 'instruct']):
            bonus += 0.3
            
        # Bonus for roleplay models (excellent for taking debate positions/personas)
        if any(word in model_name_lower for word in ['roleplay', 'character', 'persona']):
            bonus += 0.2
            
        # Bonus for uncensored models (may be more willing to take strong positions)
        if 'uncensored' in model_name_lower:
            bonus += 0.15
            
        # Penalty for base/raw models (usually not fine-tuned for conversation)
        if 'base' in model_name_lower and 'instruct' not in model_name_lower:
            bonus -= 0.3
        
        # Ensure bonus stays in reasonable range
        return max(0.5, min(bonus, 2.0))


class OpenRouterModelFilter:
    """Intelligent filtering for OpenRouter models."""
    
    # DEPRECATED: Use OpenRouterCapabilityExtractor instead
    # Known model families and their characteristics
    MODEL_FAMILIES = {
        # Meta/LLaMA family
        r'llama.*3\.1.*405b': {'weight_class': ModelWeightClass.ULTRAWEIGHT, 'params': '405B'},
        r'llama.*3\.1.*70b': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': '70B'}, 
        r'llama.*3\.1.*8b': {'weight_class': ModelWeightClass.MIDDLEWEIGHT, 'params': '8B'},
        r'llama.*3.*70b': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': '70B'},
        r'llama.*3.*8b': {'weight_class': ModelWeightClass.MIDDLEWEIGHT, 'params': '8B'},
        r'llama.*2.*70b': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': '70B'},
        r'llama.*2.*7b': {'weight_class': ModelWeightClass.MIDDLEWEIGHT, 'params': '7B'},
        
        # OpenAI
        r'gpt-4o': {'weight_class': ModelWeightClass.ULTRAWEIGHT, 'params': 'Unknown'},
        r'gpt-4.*turbo': {'weight_class': ModelWeightClass.ULTRAWEIGHT, 'params': 'Unknown'},
        r'gpt-4': {'weight_class': ModelWeightClass.ULTRAWEIGHT, 'params': 'Unknown'},
        r'gpt-3\.5.*turbo': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': 'Unknown'},
        
        # Anthropic Claude
        r'claude-3.*opus': {'weight_class': ModelWeightClass.ULTRAWEIGHT, 'params': 'Unknown'},
        r'claude-3.*sonnet': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': 'Unknown'},
        r'claude-3.*haiku': {'weight_class': ModelWeightClass.MIDDLEWEIGHT, 'params': 'Unknown'},
        
        # Google
        r'gemini.*pro': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': 'Unknown'},
        r'gemini.*flash': {'weight_class': ModelWeightClass.MIDDLEWEIGHT, 'params': 'Unknown'},
        
        # Mistral
        r'mistral.*large': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': '~70B'},
        r'mistral.*medium': {'weight_class': ModelWeightClass.MIDDLEWEIGHT, 'params': '~22B'},
        r'mistral.*small': {'weight_class': ModelWeightClass.LIGHTWEIGHT, 'params': '~7B'},
        r'mixtral.*8x7b': {'weight_class': ModelWeightClass.HEAVYWEIGHT, 'params': '8x7B'},
        r'mixtral.*8x22b': {'weight_class': ModelWeightClass.ULTRAWEIGHT, 'params': '8x22B'},
    }
    
    # Preview/anonymous model patterns
    PREVIEW_PATTERNS = [
        r'^sonoma-', r'^preview-', r'^anonymous-', r'^beta-', r'^test-',
        r'-preview$', r'-beta$', r'-alpha$', r'experimental'
    ]
    
    # Models to exclude (known problematic or irrelevant for debate/conversation)
    EXCLUDE_PATTERNS = [
        # Audio/Speech models
        r'whisper', r'tts-', r'stt-', r'bark', r'speech', r'audio',
        # Image/Vision GENERATION models (but allow multimodal INPUT models like GPT-4V, Claude-3)
        r'dall-e', r'stable-diffusion', r'midjourney', r'flux', r'sdxl', r'imagen', r'firefly',
        # Video models
        r'video', r'runway', r'luma', r'pika', r'kling',
        # Code-specific models (too specialized for general debate)
        r'code-', r'-code', r'codestral', r'codegeex', r'codegen', r'starcoder',
        r'deepseek-coder', r'wizardcoder', r'phind-code',
        # Embedding/Vector models
        r'embed', r'embedding', r'vector', r'similarity',
        # Moderation/Safety models
        r'moderation', r'safety', r'guardrail',
        # Search/RAG specific models
        r'search', r'retrieval', r'rag-',
        # Math/Science specialized models (too narrow for general debate)
        r'math-', r'scientific-', r'theorem', r'proof-',
        # Translation-only models (too specific)
        r'translate-', r'translation-', r'-translator',
        # Jailbreak variants (truly problematic models)
        r'jailbreak'
    ]
    
    @classmethod
    def is_suitable_for_debate(cls, model: OpenRouterModel) -> bool:
        """Check if model is suitable for text-based conversation and debate."""
        # Must have text input and output
        has_text_input = 'text' in model.architecture.input_modalities
        has_text_output = 'text' in model.architecture.output_modalities
        
        if not (has_text_input and has_text_output):
            return False
        
        # Exclude models that output non-text (images, audio, video)
        # These are fundamentally different from multimodal models that can INPUT multiple types
        non_text_outputs = [m for m in model.architecture.output_modalities if m != 'text']
        if len(non_text_outputs) > 0:
            return False
            
        # Allow multimodal INPUT models - they're often better at reasoning and text generation
        # Examples: GPT-4V, Claude 3, Gemini Pro Vision - excellent for debates even with image capability
        
        # Additional checks based on model name/description for debate suitability
        model_name_lower = model.name.lower()
        model_id_lower = model.id.lower()
        description_lower = model.description.lower()
        
        # Exclude models with names suggesting non-conversational focus
        non_conversational_keywords = [
            'instruct-only', 'completion-only', 'base-model', 'fine-tuned-only',
            'retrieval', 'embedding', 'classification'
        ]
        
        for keyword in non_conversational_keywords:
            if keyword in model_name_lower or keyword in description_lower:
                return False
        
        return True
    
    @classmethod
    def is_preview_model(cls, model: OpenRouterModel) -> bool:
        """Detect preview/anonymous models."""
        model_id_lower = model.id.lower()
        model_name_lower = model.name.lower()
        description_lower = model.description.lower()
        
        # Check name patterns
        for pattern in cls.PREVIEW_PATTERNS:
            if re.search(pattern, model_id_lower) or re.search(pattern, model_name_lower):
                return True
        
        # Check description for preview keywords
        preview_keywords = ['preview', 'anonymous', 'beta', 'experimental', 'temporary']
        if any(keyword in description_lower for keyword in preview_keywords):
            return True
        
        # Free models with suspicious names (don't match known providers)
        if model.pricing.is_free:
            known_free_providers = ['meta-llama', 'microsoft', 'google', 'huggingface']
            if not any(provider in model_id_lower for provider in known_free_providers):
                # Could be a preview model
                return True
        
        return False
    
    @classmethod
    def should_exclude_model(cls, model: OpenRouterModel) -> bool:
        """Check if model should be excluded entirely."""
        model_id_lower = model.id.lower()
        
        for pattern in cls.EXCLUDE_PATTERNS:
            if re.search(pattern, model_id_lower):
                return True
        
        return False
    
    @classmethod
    def classify_model(cls, model: OpenRouterModel) -> tuple[ModelWeightClass, Optional[str]]:
        """Classify model into weight class and estimate parameters using dynamic extraction."""
        
        # Use the new capability extractor for automatic classification
        estimated_params = OpenRouterCapabilityExtractor.extract_parameter_count(model)
        weight_class = OpenRouterCapabilityExtractor.determine_weight_class(model, estimated_params)
        
        return weight_class, estimated_params
    
    @classmethod
    def calculate_value_score(cls, model: OpenRouterModel) -> float:
        """Calculate value score (higher = better value) using debate-focused scoring."""
        
        # Use the new debate-focused scoring system
        weight_class = OpenRouterCapabilityExtractor.determine_weight_class(model)
        return OpenRouterCapabilityExtractor.calculate_debate_score(model, weight_class)
    
    @classmethod
    def determine_tier(cls, model: OpenRouterModel, value_score: float, weight_class: ModelWeightClass) -> ModelTier:
        """Determine model tier based on debate performance, cost efficiency, and capabilities."""
        
        if cls.is_preview_model(model):
            return ModelTier.BUDGET  # Preview models are always budget tier
        
        avg_cost = model.pricing.avg_cost_per_1k
        context_length = model.context_length
        
        # Tier determination based on debate suitability
        # Consider: context length (debate flow), cost efficiency, and model size
        
        # Free models are automatically good value
        if model.pricing.is_free:
            if weight_class == ModelWeightClass.ULTRAWEIGHT:
                return ModelTier.FLAGSHIP
            elif weight_class == ModelWeightClass.HEAVYWEIGHT:
                return ModelTier.PREMIUM
            else:
                return ModelTier.BALANCED
        
        # Cost-based tiers with context length consideration
        if weight_class == ModelWeightClass.ULTRAWEIGHT:
            # Large models: Premium if reasonably priced, Flagship if expensive but capable
            if avg_cost <= 0.01:
                return ModelTier.PREMIUM  # Great large model at good price
            elif avg_cost <= 0.02:
                return ModelTier.FLAGSHIP if context_length >= 64000 else ModelTier.PREMIUM
            else:
                return ModelTier.FLAGSHIP  # Expensive but most capable
                
        elif weight_class == ModelWeightClass.HEAVYWEIGHT:
            # Medium-large models: Good balance for debates
            if avg_cost <= 0.003:
                return ModelTier.PREMIUM  # Excellent value
            elif avg_cost <= 0.008:
                return ModelTier.BALANCED  # Good value
            else:
                return ModelTier.PREMIUM  # Higher cost but good capabilities
                
        elif weight_class == ModelWeightClass.MIDDLEWEIGHT:
            # Medium models: Depend heavily on cost
            if avg_cost <= 0.002:
                return ModelTier.BALANCED  # Good value for medium model
            elif avg_cost <= 0.006:
                return ModelTier.PREMIUM if context_length >= 32000 else ModelTier.BALANCED
            else:
                return ModelTier.PREMIUM  # Expensive medium model, likely high quality
                
        else:  # LIGHTWEIGHT
            # Small models: Mainly for budget tier unless very cheap
            if avg_cost <= 0.001:
                return ModelTier.BALANCED if context_length >= 16000 else ModelTier.BUDGET
            else:
                return ModelTier.BUDGET
    
    @classmethod
    def filter_and_enhance_models(
        cls, 
        models: List[OpenRouterModel],
        include_preview: bool = False,
        max_cost_per_1k: float = 0.02,
        min_context_length: int = 4096,
        max_models_per_tier: int = 5
    ) -> List[OpenRouterEnhancedModelInfo]:
        """Filter and enhance models with intelligent selection."""
        enhanced_models = []
        
        for model in models:
            # Skip excluded models
            if cls.should_exclude_model(model):
                continue
            
            # Skip models not suitable for debate
            if not cls.is_suitable_for_debate(model):
                continue
            
            # Skip preview models if not requested
            is_preview = cls.is_preview_model(model)
            if is_preview and not include_preview:
                continue
            
            # Skip models that are too expensive
            if model.pricing.avg_cost_per_1k > max_cost_per_1k and not model.pricing.is_free:
                continue
            
            # Skip models with insufficient context
            if model.context_length < min_context_length:
                continue
            
            # Classify and score the model
            weight_class, estimated_params = cls.classify_model(model)
            value_score = cls.calculate_value_score(model)
            tier = cls.determine_tier(model, value_score, weight_class)
            
            # Convert OpenRouter pricing to generic pricing
            generic_pricing = ModelPricing(
                prompt_cost_per_1k=model.pricing.prompt_cost_per_1k,
                completion_cost_per_1k=model.pricing.completion_cost_per_1k,
                is_free=model.pricing.is_free,
                currency="USD"
            )
            
            enhanced_model = OpenRouterEnhancedModelInfo(
                id=model.id,
                name=model.name,
                provider='openrouter',
                description=model.description,
                weight_class=weight_class,
                tier=tier,
                context_length=model.context_length,
                max_completion_tokens=model.top_provider.max_completion_tokens or 4096,
                pricing=generic_pricing,
                value_score=value_score,
                is_preview=is_preview,
                is_text_only=(len([m for m in model.architecture.input_modalities if m != 'text']) == 0),
                estimated_params=estimated_params,
                source_info={'openrouter_raw': model.dict()}
            )
            
            enhanced_models.append(enhanced_model)
        
        # Sort models by tier and value
        enhanced_models.sort(key=lambda m: m.sort_key)
        
        # Limit models per tier to avoid overwhelming users
        if max_models_per_tier > 0:
            tier_counts = {}
            filtered_models = []
            
            for model in enhanced_models:
                tier_count = tier_counts.get(model.tier, 0)
                if tier_count < max_models_per_tier:
                    filtered_models.append(model)
                    tier_counts[model.tier] = tier_count + 1
            
            enhanced_models = filtered_models
        
        return enhanced_models