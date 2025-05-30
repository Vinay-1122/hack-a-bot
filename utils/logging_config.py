import logging
import logging.config
import sys
from typing import Dict, Any
import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO", json_logs: bool = True) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs in JSON format
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create formatter
    if json_logs:
        formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={'asctime': 'timestamp', 'levelname': 'level'}
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        format='%(message)s' if json_logs else None
    )
    
    # Apply formatter to all handlers
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('docker').setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        structlog.BoundLogger: Configured logger instance
    """
    return structlog.get_logger(name)


def log_llm_interaction(
    logger: structlog.BoundLogger,
    provider: str,
    model_id: str,
    operation: str,
    prompt_length: int,
    response_length: int,
    token_usage: Dict[str, int] = None,
    duration_ms: float = None,
    success: bool = True,
    error: str = None
) -> None:
    """
    Log LLM interaction with standardized fields.
    
    Args:
        logger: Logger instance
        provider: LLM provider name
        model_id: Model ID used
        operation: Type of operation (generation, validation)
        prompt_length: Length of input prompt
        response_length: Length of response
        token_usage: Token usage information
        duration_ms: Duration in milliseconds
        success: Whether operation succeeded
        error: Error message if failed
    """
    log_data = {
        "event": "llm_interaction",
        "provider": provider,
        "model_id": model_id,
        "operation": operation,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "success": success
    }
    
    if token_usage:
        log_data.update({
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0)
        })
    
    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms
    
    if error:
        log_data["error"] = error
    
    if success:
        logger.info("LLM interaction completed", **log_data)
    else:
        logger.error("LLM interaction failed", **log_data)


def log_code_execution(
    logger: structlog.BoundLogger,
    job_id: str,
    code_length: int,
    execution_time_ms: float,
    memory_usage_mb: float = None,
    success: bool = True,
    error: str = None,
    output_size_bytes: int = None
) -> None:
    """
    Log code execution with standardized fields.
    
    Args:
        logger: Logger instance
        job_id: Job identifier
        code_length: Length of executed code
        execution_time_ms: Execution time in milliseconds
        memory_usage_mb: Memory usage in MB
        success: Whether execution succeeded
        error: Error message if failed
        output_size_bytes: Size of output in bytes
    """
    log_data = {
        "event": "code_execution",
        "job_id": job_id,
        "code_length": code_length,
        "execution_time_ms": execution_time_ms,
        "success": success
    }
    
    if memory_usage_mb is not None:
        log_data["memory_usage_mb"] = memory_usage_mb
    
    if output_size_bytes is not None:
        log_data["output_size_bytes"] = output_size_bytes
    
    if error:
        log_data["error"] = error
    
    if success:
        logger.info("Code execution completed", **log_data)
    else:
        logger.error("Code execution failed", **log_data)


def log_security_violation(
    logger: structlog.BoundLogger,
    job_id: str,
    violation_type: str,
    violation_message: str,
    code_snippet: str = None,
    line_number: int = None
) -> None:
    """
    Log security violations.
    
    Args:
        logger: Logger instance
        job_id: Job identifier
        violation_type: Type of security violation
        violation_message: Description of violation
        code_snippet: Relevant code snippet
        line_number: Line number of violation
    """
    log_data = {
        "event": "security_violation",
        "job_id": job_id,
        "violation_type": violation_type,
        "violation_message": violation_message
    }
    
    if code_snippet:
        log_data["code_snippet"] = code_snippet
    
    if line_number:
        log_data["line_number"] = line_number
    
    logger.warning("Security violation detected", **log_data) 