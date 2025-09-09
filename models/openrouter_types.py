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
        return self.prompt == "0" and self.completion == "0"


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


class OpenRouterModelFilter:
    """Intelligent filtering for OpenRouter models."""
    
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
    
    # Models to exclude (known problematic or irrelevant)
    EXCLUDE_PATTERNS = [
        r'whisper', r'dall-e', r'tts-', r'stt-', r'vision', r'embed',
        r'moderation', r'search'
    ]
    
    @classmethod
    def is_text_only_model(cls, model: OpenRouterModel) -> bool:
        """Check if model is text-only (no image/audio/video)."""
        # Must have text input and output
        has_text_input = 'text' in model.architecture.input_modalities
        has_text_output = 'text' in model.architecture.output_modalities
        
        # Exclude if it has other modalities as primary focus
        non_text_inputs = [m for m in model.architecture.input_modalities if m != 'text']
        non_text_outputs = [m for m in model.architecture.output_modalities if m != 'text']
        
        # Allow models that can take images as input but focus on text
        # Exclude models that output non-text or are primarily multimodal
        return (has_text_input and has_text_output and 
                len(non_text_outputs) == 0)
    
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
        """Classify model into weight class and estimate parameters."""
        model_id_lower = model.id.lower()
        
        for pattern, info in cls.MODEL_FAMILIES.items():
            if re.search(pattern, model_id_lower):
                return info['weight_class'], info['params']
        
        # Fallback based on context length and pricing
        if model.pricing.avg_cost_per_1k > 0.02:  # Very expensive
            return ModelWeightClass.ULTRAWEIGHT, 'Unknown'
        elif model.pricing.avg_cost_per_1k > 0.005:  # Expensive
            return ModelWeightClass.HEAVYWEIGHT, 'Unknown'
        elif model.pricing.avg_cost_per_1k > 0.001:  # Moderate
            return ModelWeightClass.MIDDLEWEIGHT, 'Unknown'
        else:  # Cheap or free
            return ModelWeightClass.LIGHTWEIGHT, 'Unknown'
    
    @classmethod
    def calculate_value_score(cls, model: OpenRouterModel) -> float:
        """Calculate value score (higher = better value)."""
        # Base score from context length
        context_score = min(model.context_length / 32768, 2.0)  # Cap at 2x for 32K+ context
        
        # Cost penalty (lower cost = higher score)
        if model.pricing.is_free:
            cost_score = 10.0  # High score for free models
        else:
            # Inverse relationship: cheaper = higher score
            avg_cost = max(model.pricing.avg_cost_per_1k, 0.0001)  # Avoid division by zero
            cost_score = 0.01 / avg_cost  # Normalize around $0.01/1K tokens
        
        # Quality bonus for known good models
        quality_bonus = 1.0
        model_id_lower = model.id.lower()
        
        if any(pattern in model_id_lower for pattern in ['gpt-4', 'claude-3', 'gemini-pro']):
            quality_bonus = 2.0
        elif any(pattern in model_id_lower for pattern in ['llama-3.1', 'mixtral']):
            quality_bonus = 1.5
        elif any(pattern in model_id_lower for pattern in ['llama-3', 'mistral']):
            quality_bonus = 1.3
        
        return context_score * cost_score * quality_bonus
    
    @classmethod
    def determine_tier(cls, model: OpenRouterModel, value_score: float, weight_class: ModelWeightClass) -> ModelTier:
        """Determine model tier based on cost and quality."""
        if cls.is_preview_model(model):
            return ModelTier.BUDGET  # Preview models are budget tier
        
        avg_cost = model.pricing.avg_cost_per_1k
        
        if weight_class == ModelWeightClass.ULTRAWEIGHT:
            if avg_cost > 0.015:
                return ModelTier.FLAGSHIP
            else:
                return ModelTier.PREMIUM
        elif weight_class == ModelWeightClass.HEAVYWEIGHT:
            if avg_cost > 0.01:
                return ModelTier.PREMIUM
            elif avg_cost > 0.002:
                return ModelTier.BALANCED
            else:
                return ModelTier.BUDGET
        elif weight_class == ModelWeightClass.MIDDLEWEIGHT:
            if avg_cost > 0.005:
                return ModelTier.PREMIUM
            elif avg_cost > 0.001:
                return ModelTier.BALANCED
            else:
                return ModelTier.BUDGET
        else:  # LIGHTWEIGHT
            if avg_cost > 0.001:
                return ModelTier.BALANCED
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
            
            # Skip non-text models
            if not cls.is_text_only_model(model):
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
                is_text_only=True,
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