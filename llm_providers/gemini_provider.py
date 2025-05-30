import logging
from typing import Dict, Any, Optional
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import GenerateContentResponse

from .llm_provider_interface import AbstractLLMProvider, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)


class GoogleGeminiProvider(AbstractLLMProvider):
    """Google Gemini LLM provider implementation."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "gemini"
        self.model = None
        
        # Supported Gemini models
        self.supported_models = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    
    def initialize_client(self, config: Dict[str, Any]) -> None:
        """Initialize Gemini client with API key."""
        try:
            api_key = config.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for Gemini provider")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Test the connection with a simple model instantiation
            test_model = genai.GenerativeModel('gemini-pro')
            
            self.is_initialized = True
            logger.info("Gemini provider initialized successfully")
            
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Invalid Gemini API key: {e}")
            raise ValueError(f"Invalid Gemini API key: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise ConnectionError(f"Failed to initialize Gemini provider: {e}")
    
    def generate_text(
        self, 
        prompt: str, 
        model_id: str, 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate text using Gemini."""
        if not self.is_initialized:
            raise RuntimeError("Gemini provider not initialized")
        
        if not self.validate_model_id(model_id):
            raise ValueError(f"Unsupported model ID: {model_id}")
        
        config = config or LLMConfig()
        
        try:
            # Create model instance
            model = genai.GenerativeModel(model_id)
            
            # Prepare generation config
            generation_config = genai.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                top_p=config.top_p,
                top_k=config.top_k
            )
            
            # Generate content
            logger.info(f"Generating text with Gemini model: {model_id}")
            response: GenerateContentResponse = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if response.text:
                generated_text = response.text
            else:
                raise RuntimeError("Empty response from Gemini")
            
            # Extract token usage if available
            token_usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            # Extract metadata
            metadata = {
                "finish_reason": getattr(response.candidates[0] if response.candidates else None, 'finish_reason', None),
                "safety_ratings": [
                    {
                        "category": rating.category.name,
                        "probability": rating.probability.name
                    } for rating in (response.candidates[0].safety_ratings if response.candidates else [])
                ]
            }
            
            logger.info(f"Successfully generated text with Gemini (tokens: {token_usage})")
            
            return LLMResponse(
                text=generated_text,
                model_id=model_id,
                provider=self.provider_name,
                token_usage=token_usage,
                metadata=metadata
            )
            
        except google_exceptions.ResourceExhausted as e:
            error_msg = f"Gemini API quota exceeded: {e}"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model_id=model_id,
                provider=self.provider_name,
                error=error_msg
            )
            
        except google_exceptions.InvalidArgument as e:
            error_msg = f"Invalid argument for Gemini API: {e}"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model_id=model_id,
                provider=self.provider_name,
                error=error_msg
            )
            
        except google_exceptions.PermissionDenied as e:
            error_msg = f"Permission denied for Gemini API: {e}"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model_id=model_id,
                provider=self.provider_name,
                error=error_msg
            )
            
        except Exception as e:
            error_msg = f"Unexpected error with Gemini: {e}"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model_id=model_id,
                provider=self.provider_name,
                error=error_msg
            )
    
    def validate_model_id(self, model_id: str) -> bool:
        """Validate if model ID is supported by Gemini."""
        return model_id in self.supported_models
    
    def get_supported_models(self) -> list[str]:
        """Get list of supported Gemini models."""
        return self.supported_models.copy()
    
    def health_check(self) -> bool:
        """Perform health check for Gemini provider."""
        if not self.is_initialized:
            return False
        
        try:
            # Simple test generation
            test_response = self.generate_text(
                prompt="Test",
                model_id="gemini-pro",
                config=LLMConfig(temperature=0.1, max_output_tokens=5)
            )
            return test_response.error is None and len(test_response.text) > 0
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False 