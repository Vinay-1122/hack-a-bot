import json
import logging
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .llm_provider_interface import AbstractLLMProvider, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)


class AWSBedrockProvider(AbstractLLMProvider):
    """AWS Bedrock LLM provider implementation."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "bedrock"
        self.client = None
        
        # Supported Bedrock models
        self.supported_models = [
            # Anthropic Claude models
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-v2:1",
            "anthropic.claude-v2",
            "anthropic.claude-instant-v1",
            
            # Amazon Titan models
            "amazon.titan-text-express-v1",
            "amazon.titan-text-lite-v1",
            
            # Meta Llama models
            "meta.llama2-70b-chat-v1",
            "meta.llama2-13b-chat-v1",
            
            # Cohere models
            "cohere.command-text-v14",
            "cohere.command-light-text-v14"
        ]
    
    def initialize_client(self, config: Dict[str, Any]) -> None:
        """Initialize Bedrock client with AWS credentials."""
        try:
            region = config.get("region", "us-east-1")
            access_key = config.get("access_key_id")
            secret_key = config.get("secret_access_key")
            
            # Initialize boto3 client
            if access_key and secret_key:
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=region,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key
                )
            else:
                # Use default credentials (IAM role, environment variables, etc.)
                self.client = boto3.client('bedrock-runtime', region_name=region)
            
            # Test the connection
            self.client.list_foundation_models()
            
            self.is_initialized = True
            logger.info(f"Bedrock provider initialized successfully in region: {region}")
            
        except NoCredentialsError:
            error_msg = "AWS credentials not found for Bedrock provider"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except ClientError as e:
            error_msg = f"Failed to initialize Bedrock client: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error initializing Bedrock provider: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def generate_text(
        self, 
        prompt: str, 
        model_id: str, 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate text using Bedrock."""
        if not self.is_initialized:
            raise RuntimeError("Bedrock provider not initialized")
        
        if not self.validate_model_id(model_id):
            raise ValueError(f"Unsupported model ID: {model_id}")
        
        config = config or LLMConfig()
        
        try:
            # Prepare request body based on model type
            request_body = self._prepare_request_body(prompt, model_id, config)
            
            logger.info(f"Generating text with Bedrock model: {model_id}")
            
            # Call Bedrock
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read().decode('utf-8'))
            generated_text, token_usage, metadata = self._parse_response(response_body, model_id)
            
            logger.info(f"Successfully generated text with Bedrock (tokens: {token_usage})")
            
            return LLMResponse(
                text=generated_text,
                model_id=model_id,
                provider=self.provider_name,
                token_usage=token_usage,
                metadata=metadata
            )
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = f"Bedrock API error ({error_code}): {e}"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model_id=model_id,
                provider=self.provider_name,
                error=error_msg
            )
            
        except Exception as e:
            error_msg = f"Unexpected error with Bedrock: {e}"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model_id=model_id,
                provider=self.provider_name,
                error=error_msg
            )
    
    def _prepare_request_body(self, prompt: str, model_id: str, config: LLMConfig) -> Dict[str, Any]:
        """Prepare request body based on model type."""
        if model_id.startswith("anthropic.claude"):
            # Claude models format
            return {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "temperature": config.temperature,
                "max_tokens_to_sample": config.max_output_tokens,
                "top_p": config.top_p,
                "top_k": config.top_k
            }
        elif model_id.startswith("amazon.titan"):
            # Titan models format
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "temperature": config.temperature,
                    "maxTokenCount": config.max_output_tokens,
                    "topP": config.top_p
                }
            }
        elif model_id.startswith("meta.llama"):
            # Llama models format
            return {
                "prompt": prompt,
                "temperature": config.temperature,
                "max_gen_len": config.max_output_tokens,
                "top_p": config.top_p
            }
        elif model_id.startswith("cohere.command"):
            # Cohere models format
            return {
                "prompt": prompt,
                "temperature": config.temperature,
                "max_tokens": config.max_output_tokens,
                "p": config.top_p,
                "k": config.top_k
            }
        else:
            # Default format (Claude-like)
            return {
                "prompt": prompt,
                "temperature": config.temperature,
                "max_tokens_to_sample": config.max_output_tokens,
                "top_p": config.top_p,
                "top_k": config.top_k
            }
    
    def _parse_response(self, response_body: Dict[str, Any], model_id: str) -> tuple[str, Dict[str, int], Dict[str, Any]]:
        """Parse response body based on model type."""
        if model_id.startswith("anthropic.claude"):
            text = response_body.get("completion", "")
            token_usage = {
                "prompt_tokens": response_body.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response_body.get("usage", {}).get("output_tokens", 0),
                "total_tokens": response_body.get("usage", {}).get("input_tokens", 0) + response_body.get("usage", {}).get("output_tokens", 0)
            }
            metadata = {
                "stop_reason": response_body.get("stop_reason"),
                "model": response_body.get("model")
            }
        elif model_id.startswith("amazon.titan"):
            text = response_body.get("results", [{}])[0].get("outputText", "")
            token_usage = {
                "prompt_tokens": response_body.get("inputTextTokenCount", 0),
                "completion_tokens": response_body.get("results", [{}])[0].get("tokenCount", 0),
                "total_tokens": response_body.get("inputTextTokenCount", 0) + response_body.get("results", [{}])[0].get("tokenCount", 0)
            }
            metadata = {
                "completionReason": response_body.get("results", [{}])[0].get("completionReason")
            }
        elif model_id.startswith("meta.llama"):
            text = response_body.get("generation", "")
            token_usage = {
                "prompt_tokens": response_body.get("prompt_token_count", 0),
                "completion_tokens": response_body.get("generation_token_count", 0),
                "total_tokens": response_body.get("prompt_token_count", 0) + response_body.get("generation_token_count", 0)
            }
            metadata = {
                "stop_reason": response_body.get("stop_reason")
            }
        elif model_id.startswith("cohere.command"):
            text = response_body.get("generations", [{}])[0].get("text", "")
            token_usage = {
                "prompt_tokens": 0,  # Cohere doesn't always provide this
                "completion_tokens": 0,  # Cohere doesn't always provide this
                "total_tokens": 0
            }
            metadata = {
                "finish_reason": response_body.get("generations", [{}])[0].get("finish_reason")
            }
        else:
            # Default parsing (Claude-like)
            text = response_body.get("completion", response_body.get("text", ""))
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            metadata = response_body
        
        return text, token_usage, metadata
    
    def validate_model_id(self, model_id: str) -> bool:
        """Validate if model ID is supported by Bedrock."""
        return model_id in self.supported_models
    
    def get_supported_models(self) -> list[str]:
        """Get list of supported Bedrock models."""
        return self.supported_models.copy()
    
    def health_check(self) -> bool:
        """Perform health check for Bedrock provider."""
        if not self.is_initialized:
            return False
        
        try:
            # Simple test generation
            test_response = self.generate_text(
                prompt="Test",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                config=LLMConfig(temperature=0.1, max_output_tokens=5)
            )
            return test_response.error is None and len(test_response.text) > 0
        except Exception as e:
            logger.error(f"Bedrock health check failed: {e}")
            return False 