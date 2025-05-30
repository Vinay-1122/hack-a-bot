import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # LLM Provider Configuration
    LLM_PROVIDER: str = Field(default="gemini", description="Primary LLM provider (gemini/bedrock)")
    
    # Gemini Configuration
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    CODE_GEN_MODEL_ID_GEMINI: str = Field(default="gemini-pro", description="Gemini model for code generation")
    CODE_VAL_MODEL_ID_GEMINI: str = Field(default="gemini-pro", description="Gemini model for code validation")
    GEMINI_TEMPERATURE: float = Field(default=0.3, description="Gemini temperature for generation")
    GEMINI_MAX_OUTPUT_TOKENS: int = Field(default=2048, description="Gemini max output tokens")
    
    # Bedrock Configuration
    BEDROCK_REGION: str = Field(default="us-east-1", description="AWS Bedrock region")
    BEDROCK_ACCESS_KEY_ID: Optional[str] = Field(default=None, description="AWS access key ID")
    BEDROCK_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, description="AWS secret access key")
    CODE_GEN_MODEL_ID_BEDROCK: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0", 
        description="Bedrock model for code generation"
    )
    CODE_VAL_MODEL_ID_BEDROCK: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0", 
        description="Bedrock model for code validation"
    )
    BEDROCK_TEMPERATURE: float = Field(default=0.3, description="Bedrock temperature for generation")
    BEDROCK_MAX_OUTPUT_TOKENS: int = Field(default=2048, description="Bedrock max output tokens")
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/plotting_service",
        description="Database connection URL"
    )
    
    # Celery Configuration
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", description="Celery broker URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", description="Celery result backend")
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, description="AWS S3 access key")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, description="AWS S3 secret key")
    S3_BUCKET_NAME: str = Field(default="plotting-service-outputs", description="S3 bucket for plot outputs")
    S3_REGION: str = Field(default="us-east-1", description="S3 region")
    
    # Docker Configuration
    DOCKER_IMAGE: str = Field(default="python:3.11-slim", description="Docker image for code execution")
    DOCKER_TIMEOUT: int = Field(default=300, description="Docker execution timeout in seconds")
    DOCKER_MEMORY_LIMIT: str = Field(default="512m", description="Docker memory limit")
    DOCKER_CPU_LIMIT: str = Field(default="1.0", description="Docker CPU limit")
    
    # Security Configuration
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", description="Application secret key")
    API_KEY_HEADER: str = Field(default="X-API-Key", description="API key header name")
    
    # Monitoring Configuration
    PROMETHEUS_PORT: int = Field(default=8001, description="Prometheus metrics port")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Execution Configuration
    MAX_CONCURRENT_EXECUTIONS: int = Field(default=10, description="Maximum concurrent code executions")
    TEMP_DIR_BASE: str = Field(default="/tmp/plotting_service", description="Base directory for temporary files")
    CLEANUP_TEMP_FILES: bool = Field(default=True, description="Whether to cleanup temporary files")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="API rate limit per minute")
    
    # Job Management
    JOB_RETENTION_HOURS: int = Field(default=24, description="Job result retention in hours")
    MAX_JOB_QUEUE_SIZE: int = Field(default=1000, description="Maximum job queue size")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_llm_config() -> dict:
    """Get LLM configuration based on the selected provider."""
    if settings.LLM_PROVIDER.lower() == "gemini":
        return {
            "provider": "gemini",
            "api_key": settings.GEMINI_API_KEY,
            "code_gen_model": settings.CODE_GEN_MODEL_ID_GEMINI,
            "code_val_model": settings.CODE_VAL_MODEL_ID_GEMINI,
            "temperature": settings.GEMINI_TEMPERATURE,
            "max_output_tokens": settings.GEMINI_MAX_OUTPUT_TOKENS,
        }
    elif settings.LLM_PROVIDER.lower() == "bedrock":
        return {
            "provider": "bedrock",
            "region": settings.BEDROCK_REGION,
            "access_key_id": settings.BEDROCK_ACCESS_KEY_ID,
            "secret_access_key": settings.BEDROCK_SECRET_ACCESS_KEY,
            "code_gen_model": settings.CODE_GEN_MODEL_ID_BEDROCK,
            "code_val_model": settings.CODE_VAL_MODEL_ID_BEDROCK,
            "temperature": settings.BEDROCK_TEMPERATURE,
            "max_output_tokens": settings.BEDROCK_MAX_OUTPUT_TOKENS,
        }
    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")


def get_redis_config() -> dict:
    """Get Redis configuration."""
    return {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "db": settings.REDIS_DB,
        "password": settings.REDIS_PASSWORD,
        "decode_responses": True,
    }


def get_s3_config() -> dict:
    """Get S3 configuration."""
    return {
        "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
        "region_name": settings.S3_REGION,
        "bucket_name": settings.S3_BUCKET_NAME,
    } 