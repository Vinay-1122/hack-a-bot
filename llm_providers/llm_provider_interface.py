from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    text: str
    model_id: str
    provider: str
    token_usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM requests."""
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 40


class AbstractLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self):
        self.client = None
        self.provider_name = ""
        self.is_initialized = False
    
    @abstractmethod
    def initialize_client(self, config: Dict[str, Any]) -> None:
        """
        Initialize the LLM client for the provider.
        
        Args:
            config: Provider-specific configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If unable to connect to the provider
        """
        pass
    
    @abstractmethod
    def generate_text(
        self, 
        prompt: str, 
        model_id: str, 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generate text using the specified model.
        
        Args:
            prompt: The input prompt for text generation
            model_id: The specific model to use for generation
            config: Optional configuration for the generation
            
        Returns:
            LLMResponse: Standardized response containing generated text and metadata
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    def validate_model_id(self, model_id: str) -> bool:
        """
        Validate if the model ID is supported by this provider.
        
        Args:
            model_id: The model ID to validate
            
        Returns:
            bool: True if model is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """
        Get list of supported model IDs for this provider.
        
        Returns:
            list[str]: List of supported model IDs
        """
        pass
    
    def health_check(self) -> bool:
        """
        Perform a health check for the provider.
        
        Returns:
            bool: True if provider is healthy, False otherwise
        """
        try:
            if not self.is_initialized:
                return False
            
            # Attempt a simple generation with minimal content
            test_response = self.generate_text(
                prompt="Hello",
                model_id=self.get_supported_models()[0],
                config=LLMConfig(temperature=0.1, max_output_tokens=10)
            )
            return test_response.error is None
        except Exception:
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the provider.
        
        Returns:
            dict: Provider information including name, status, and capabilities
        """
        return {
            "provider_name": self.provider_name,
            "is_initialized": self.is_initialized,
            "supported_models": self.get_supported_models(),
            "is_healthy": self.health_check()
        } 