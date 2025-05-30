import logging
from typing import Dict, Any
from .llm_provider_interface import AbstractLLMProvider
from .gemini_provider import GoogleGeminiProvider
from .bedrock_provider import AWSBedrockProvider

logger = logging.getLogger(__name__)


def get_llm_provider(provider_name: str, config: Dict[str, Any]) -> AbstractLLMProvider:
    """
    Factory function to instantiate the correct LLM provider.
    
    Args:
        provider_name: Name of the provider ('gemini' or 'bedrock')
        config: Configuration dictionary for the provider
        
    Returns:
        AbstractLLMProvider: Initialized provider instance
        
    Raises:
        ValueError: If provider_name is unknown
        ConnectionError: If provider initialization fails
    """
    provider_name = provider_name.lower().strip()
    
    logger.info(f"Creating LLM provider: {provider_name}")
    
    if provider_name == "gemini":
        provider = GoogleGeminiProvider()
    elif provider_name == "bedrock":
        provider = AWSBedrockProvider()
    else:
        supported_providers = ["gemini", "bedrock"]
        raise ValueError(f"Unknown LLM provider: {provider_name}. Supported providers: {supported_providers}")
    
    # Initialize the provider
    try:
        provider.initialize_client(config)
        logger.info(f"Successfully initialized {provider_name} provider")
        return provider
    except Exception as e:
        logger.error(f"Failed to initialize {provider_name} provider: {e}")
        raise


def get_available_providers() -> list[str]:
    """
    Get list of available LLM providers.
    
    Returns:
        list[str]: List of available provider names
    """
    return ["gemini", "bedrock"]


def get_provider_models(provider_name: str) -> list[str]:
    """
    Get supported models for a specific provider without initializing it.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        list[str]: List of supported model IDs
        
    Raises:
        ValueError: If provider_name is unknown
    """
    provider_name = provider_name.lower().strip()
    
    if provider_name == "gemini":
        provider = GoogleGeminiProvider()
    elif provider_name == "bedrock":
        provider = AWSBedrockProvider()
    else:
        supported_providers = ["gemini", "bedrock"]
        raise ValueError(f"Unknown LLM provider: {provider_name}. Supported providers: {supported_providers}")
    
    return provider.get_supported_models()


def validate_provider_config(provider_name: str, config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate provider configuration without full initialization.
    
    Args:
        provider_name: Name of the provider
        config: Configuration dictionary
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    provider_name = provider_name.lower().strip()
    
    try:
        if provider_name == "gemini":
            if not config.get("api_key"):
                return False, "GEMINI_API_KEY is required"
        elif provider_name == "bedrock":
            region = config.get("region")
            if not region:
                return False, "BEDROCK_REGION is required"
            # Access keys are optional if using IAM roles
        else:
            return False, f"Unknown provider: {provider_name}"
        
        return True, "Configuration is valid"
        
    except Exception as e:
        return False, f"Configuration validation error: {e}" 