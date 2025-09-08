#!/usr/bin/env python3
"""Test script for multi-provider support."""

import asyncio
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from config.settings import AppConfig
from models.manager import ModelManager
from models.providers import ProviderFactory

async def test_providers():
    """Test the multi-provider system."""
    print("🧪 Testing Multi-Provider Support")
    print("=" * 50)
    
    # Test 1: Load configuration
    print("\n1. Testing Configuration Loading...")
    try:
        config_path = Path("test_providers_config.json")
        config = AppConfig.load_from_file(config_path)
        print(f"✅ Configuration loaded successfully")
        print(f"   Models configured: {list(config.models.keys())}")
        for model_id, model_config in config.models.items():
            print(f"   - {model_id}: {model_config.name} ({model_config.provider})")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return
    
    # Test 2: Provider Factory
    print("\n2. Testing Provider Factory...")
    try:
        available_providers = ProviderFactory.get_available_providers()
        print(f"✅ Available providers: {available_providers}")
        
        for provider_name in available_providers:
            provider = ProviderFactory.create_provider(provider_name, config.system)
            print(f"   - {provider_name}: {provider.__class__.__name__}")
    except Exception as e:
        print(f"❌ Provider factory failed: {e}")
        return
    
    # Test 3: Model Manager
    print("\n3. Testing Model Manager...")
    try:
        model_manager = ModelManager(config.system)
        
        # Register models
        for model_id, model_config in config.models.items():
            try:
                model_manager.register_model(model_id, model_config)
                print(f"✅ Registered {model_id} ({model_config.provider})")
            except Exception as e:
                print(f"⚠️  Failed to register {model_id}: {e}")
        
    except Exception as e:
        print(f"❌ Model manager failed: {e}")
        return
    
    # Test 4: Get Available Models
    print("\n4. Testing Model Discovery...")
    try:
        models_by_provider = await model_manager.get_available_models()
        print("✅ Available models by provider:")
        for provider, models in models_by_provider.items():
            if models:
                print(f"   - {provider}: {len(models)} models")
                # Show first 3 models as example
                for model in models[:3]:
                    print(f"     • {model}")
                if len(models) > 3:
                    print(f"     ... and {len(models) - 3} more")
            else:
                print(f"   - {provider}: No models available (check connection/API key)")
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
    
    # Test 5: Provider Status
    print("\n5. Testing Provider Status...")
    ollama_provider = ProviderFactory.create_provider("ollama", config.system)
    openrouter_provider = ProviderFactory.create_provider("openrouter", config.system)
    
    print(f"   - Ollama: {ollama_provider.__class__.__name__}")
    print(f"   - OpenRouter: {openrouter_provider.__class__.__name__}")
    
    # Check OpenRouter API key
    import os
    or_key = config.system.openrouter.api_key or os.getenv("OPENROUTER_API_KEY")
    if or_key:
        print(f"   - OpenRouter API Key: Configured (ends with ...{or_key[-4:]})")
    else:
        print(f"   - OpenRouter API Key: Not configured (set OPENROUTER_API_KEY env var)")
    
    print("\n" + "=" * 50)
    print("🎉 Multi-provider testing complete!")
    print("\nNext steps:")
    print("- Set OPENROUTER_API_KEY environment variable to test OpenRouter models")
    print("- Run a debate with mixed providers using the web interface")
    print("- Check the web API endpoints: /api/models and /api/providers")

if __name__ == "__main__":
    asyncio.run(test_providers())