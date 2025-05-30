import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from llm_providers import get_llm_provider, AbstractLLMProvider
from llm_providers.llm_provider_interface import LLMConfig, LLMResponse
from utils.prompt_builder import PromptBuilder, DataFrameSchema, DatabaseSchema
from utils.code_analyzer import CodeAnalyzer, CodeAnalysisResult
from utils.logging_config import log_llm_interaction, log_security_violation
from utils.metrics import get_metrics, measure_llm_request
from config import get_llm_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of code generation and validation."""
    success: bool
    code: Optional[str] = None
    error: Optional[str] = None
    validation_result: Optional[CodeAnalysisResult] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    validation_metadata: Optional[Dict[str, Any]] = None


class PlotGenerator:
    """Core orchestrator for LLM-based code generation and validation."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plot generator.
        
        Args:
            llm_config: Optional LLM configuration. If None, uses global config.
        """
        self.llm_config = llm_config or get_llm_config()
        self.prompt_builder = PromptBuilder()
        self.code_analyzer = CodeAnalyzer()
        self.metrics = get_metrics()
        
        # Initialize LLM providers
        self.code_gen_provider = None
        self.code_val_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize LLM providers for code generation and validation."""
        try:
            provider_name = self.llm_config["provider"]
            
            # Initialize code generation provider
            self.code_gen_provider = get_llm_provider(provider_name, self.llm_config)
            
            # For validation, we might use a different (cheaper) model
            val_config = self.llm_config.copy()
            if "code_val_model" in val_config:
                val_config["code_gen_model"] = val_config["code_val_model"]
            
            self.code_val_provider = get_llm_provider(provider_name, val_config)
            
            logger.info(f"Initialized {provider_name} providers for generation and validation")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM providers: {e}")
            raise
    
    def generate_and_validate_code(
        self,
        user_question: str,
        dataframes: List[DataFrameSchema],
        db_schema: Optional[DatabaseSchema] = None,
        job_id: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate and validate plotting code based on user request.
        
        Args:
            user_question: User's plotting request
            dataframes: Available DataFrame schemas
            db_schema: Optional database schema
            job_id: Optional job identifier for logging
            additional_context: Additional context for generation
            
        Returns:
            GenerationResult: Complete result of generation and validation
        """
        job_id = job_id or f"job_{int(time.time())}"
        
        logger.info(f"Starting code generation for job {job_id}")
        
        try:
            # Step 1: Generate code using LLM
            generation_start = time.time()
            code_result = self._generate_code(
                user_question, dataframes, db_schema, job_id, additional_context
            )
            generation_duration = time.time() - generation_start
            
            if not code_result.success:
                self.metrics.record_code_generation("failed")
                return code_result
            
            self.metrics.record_code_generation("success")
            
            # Step 2: Perform static analysis
            static_analysis_start = time.time()
            static_result = self.code_analyzer.analyze_code(code_result.code)
            static_analysis_duration = time.time() - static_analysis_start
            
            if not static_result.is_safe:
                logger.warning(f"Static analysis failed for job {job_id}")
                self._log_security_violations(job_id, static_result, code_result.code)
                
                return GenerationResult(
                    success=False,
                    error="Code failed static security analysis",
                    validation_result=static_result,
                    generation_metadata=code_result.generation_metadata
                )
            
            self.metrics.record_code_validation("static_pass")
            
            # Step 3: LLM-based validation
            llm_validation_start = time.time()
            llm_validation_result = self._validate_with_llm(
                code_result.code, user_question, job_id
            )
            llm_validation_duration = time.time() - llm_validation_start
            
            if not llm_validation_result.success:
                self.metrics.record_code_validation("llm_fail")
                return GenerationResult(
                    success=False,
                    error=llm_validation_result.error,
                    code=code_result.code,
                    validation_result=static_result,
                    generation_metadata=code_result.generation_metadata,
                    validation_metadata=llm_validation_result.validation_metadata
                )
            
            self.metrics.record_code_validation("llm_pass")
            
            logger.info(f"Code generation and validation completed for job {job_id}")
            
            return GenerationResult(
                success=True,
                code=code_result.code,
                validation_result=static_result,
                generation_metadata=code_result.generation_metadata,
                validation_metadata=llm_validation_result.validation_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in code generation for job {job_id}: {e}")
            self.metrics.record_code_generation("error")
            return GenerationResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _generate_code(
        self,
        user_question: str,
        dataframes: List[DataFrameSchema],
        db_schema: Optional[DatabaseSchema],
        job_id: str,
        additional_context: Optional[str]
    ) -> GenerationResult:
        """Generate code using LLM."""
        try:
            # Build prompt
            prompt = self.prompt_builder.build_code_generation_prompt(
                user_question=user_question,
                dataframes=dataframes,
                db_schema=db_schema,
                provider=self.llm_config["provider"],
                additional_context=additional_context
            )
            
            # Configure LLM
            llm_config = LLMConfig(
                temperature=self.llm_config.get("temperature", 0.3),
                max_output_tokens=self.llm_config.get("max_output_tokens", 2048)
            )
            
            # Generate with metrics
            with measure_llm_request(
                self.llm_config["provider"],
                self.llm_config["code_gen_model"],
                "generation"
            ):
                response = self.code_gen_provider.generate_text(
                    prompt=prompt,
                    model_id=self.llm_config["code_gen_model"],
                    config=llm_config
                )
            
            # Log the interaction
            log_llm_interaction(
                logger=logger,
                provider=response.provider,
                model_id=response.model_id,
                operation="generation",
                prompt_length=len(prompt),
                response_length=len(response.text),
                token_usage=response.token_usage,
                success=response.error is None,
                error=response.error
            )
            
            if response.error:
                return GenerationResult(
                    success=False,
                    error=f"LLM generation failed: {response.error}",
                    generation_metadata={"response": response.metadata}
                )
            
            # Extract code from response
            code = self.code_analyzer.extract_code_from_response(response.text)
            if not code:
                return GenerationResult(
                    success=False,
                    error="No valid Python code found in LLM response",
                    generation_metadata={
                        "response": response.metadata,
                        "raw_response": response.text[:500]  # Truncated for logging
                    }
                )
            
            return GenerationResult(
                success=True,
                code=code,
                generation_metadata={
                    "model_id": response.model_id,
                    "provider": response.provider,
                    "token_usage": response.token_usage,
                    "prompt_length": len(prompt),
                    "response_length": len(response.text)
                }
            )
            
        except Exception as e:
            logger.error(f"Code generation error for job {job_id}: {e}")
            return GenerationResult(
                success=False,
                error=f"Code generation failed: {str(e)}"
            )
    
    def _validate_with_llm(
        self,
        code: str,
        context: str,
        job_id: str
    ) -> GenerationResult:
        """Validate code using LLM."""
        try:
            # Build validation prompt
            prompt = self.prompt_builder.build_code_validation_prompt(
                code=code,
                context=context,
                provider=self.llm_config["provider"]
            )
            
            # Configure LLM for validation
            llm_config = LLMConfig(
                temperature=0.1,  # Lower temperature for validation
                max_output_tokens=512  # Shorter response for validation
            )
            
            # Validate with metrics
            with measure_llm_request(
                self.llm_config["provider"],
                self.llm_config.get("code_val_model", self.llm_config["code_gen_model"]),
                "validation"
            ):
                response = self.code_val_provider.generate_text(
                    prompt=prompt,
                    model_id=self.llm_config.get("code_val_model", self.llm_config["code_gen_model"]),
                    config=llm_config
                )
            
            # Log the interaction
            log_llm_interaction(
                logger=logger,
                provider=response.provider,
                model_id=response.model_id,
                operation="validation",
                prompt_length=len(prompt),
                response_length=len(response.text),
                token_usage=response.token_usage,
                success=response.error is None,
                error=response.error
            )
            
            if response.error:
                return GenerationResult(
                    success=False,
                    error=f"LLM validation failed: {response.error}",
                    validation_metadata={"response": response.metadata}
                )
            
            # Parse validation response
            validation_result = self._parse_validation_response(response.text, job_id)
            
            return GenerationResult(
                success=validation_result["is_safe"],
                error=None if validation_result["is_safe"] else validation_result.get("reason", "LLM marked code as unsafe"),
                validation_metadata={
                    "model_id": response.model_id,
                    "provider": response.provider,
                    "token_usage": response.token_usage,
                    "validation_result": validation_result
                }
            )
            
        except Exception as e:
            logger.error(f"LLM validation error for job {job_id}: {e}")
            return GenerationResult(
                success=False,
                error=f"LLM validation failed: {str(e)}"
            )
    
    def _parse_validation_response(self, response_text: str, job_id: str) -> Dict[str, Any]:
        """Parse LLM validation response."""
        try:
            # Try to extract JSON from response
            import re
            json_pattern = r'\{[^{}]*"is_safe"[^{}]*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if matches:
                return json.loads(matches[0])
            
            # Fallback: look for explicit safe/unsafe indicators
            response_lower = response_text.lower()
            if "is_safe: true" in response_lower or '"is_safe": true' in response_lower:
                return {"is_safe": True, "reason": "LLM marked as safe"}
            elif "is_safe: false" in response_lower or '"is_safe": false' in response_lower:
                return {"is_safe": False, "reason": "LLM marked as unsafe"}
            
            # Default to unsafe if unclear
            logger.warning(f"Could not parse validation response for job {job_id}: {response_text[:100]}")
            return {"is_safe": False, "reason": "Could not parse validation response"}
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in validation response for job {job_id}")
            return {"is_safe": False, "reason": "Invalid validation response format"}
    
    def _log_security_violations(
        self,
        job_id: str,
        analysis_result: CodeAnalysisResult,
        code: str
    ) -> None:
        """Log security violations for monitoring."""
        for violation in analysis_result.violations:
            log_security_violation(
                logger=logger,
                job_id=job_id,
                violation_type=violation.type,
                violation_message=violation.message,
                line_number=violation.line_number
            )
            
            # Record metrics
            self.metrics.record_security_violation(violation.type)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            "healthy": True,
            "components": {},
            "timestamp": time.time()
        }
        
        # Check code generation provider
        try:
            gen_healthy = self.code_gen_provider.health_check()
            health_status["components"]["code_gen_provider"] = {
                "healthy": gen_healthy,
                "provider": self.code_gen_provider.provider_name
            }
            if not gen_healthy:
                health_status["healthy"] = False
        except Exception as e:
            health_status["components"]["code_gen_provider"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["healthy"] = False
        
        # Check code validation provider
        try:
            val_healthy = self.code_val_provider.health_check()
            health_status["components"]["code_val_provider"] = {
                "healthy": val_healthy,
                "provider": self.code_val_provider.provider_name
            }
            if not val_healthy:
                health_status["healthy"] = False
        except Exception as e:
            health_status["components"]["code_val_provider"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["healthy"] = False
        
        return health_status 