import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
import threading


class MetricsCollector:
    """Prometheus metrics collector for the plotting service."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        
        # LLM Interaction Metrics
        self.llm_requests_total = Counter(
            'llm_requests_total',
            'Total number of LLM requests',
            ['provider', 'model_id', 'operation'],
            registry=self.registry
        )
        
        self.llm_request_duration = Histogram(
            'llm_request_duration_seconds',
            'Duration of LLM requests in seconds',
            ['provider', 'model_id', 'operation'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.llm_tokens_used = Counter(
            'llm_tokens_used_total',
            'Total tokens used by LLM',
            ['provider', 'model_id', 'token_type'],
            registry=self.registry
        )
        
        self.llm_errors_total = Counter(
            'llm_errors_total',
            'Total number of LLM errors',
            ['provider', 'model_id', 'error_type'],
            registry=self.registry
        )
        
        # Code Generation and Validation Metrics
        self.code_generation_total = Counter(
            'code_generation_requests_total',
            'Total number of code generation requests',
            ['status'],
            registry=self.registry
        )
        
        self.code_validation_total = Counter(
            'code_validation_requests_total',
            'Total number of code validation requests',
            ['result'],
            registry=self.registry
        )
        
        self.security_violations_total = Counter(
            'security_violations_total',
            'Total number of security violations detected',
            ['violation_type'],
            registry=self.registry
        )
        
        # Code Execution Metrics
        self.code_execution_total = Counter(
            'code_execution_requests_total',
            'Total number of code execution requests',
            ['status'],
            registry=self.registry
        )
        
        self.code_execution_duration = Histogram(
            'code_execution_duration_seconds',
            'Duration of code execution in seconds',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self.registry
        )
        
        self.docker_memory_usage = Histogram(
            'docker_memory_usage_mb',
            'Memory usage during code execution in MB',
            buckets=[64, 128, 256, 512, 1024, 2048],
            registry=self.registry
        )
        
        # Job Management Metrics
        self.jobs_total = Counter(
            'jobs_total',
            'Total number of jobs processed',
            ['status'],
            registry=self.registry
        )
        
        self.job_queue_size = Gauge(
            'job_queue_size',
            'Current size of job queue',
            registry=self.registry
        )
        
        self.active_executions = Gauge(
            'active_code_executions',
            'Number of currently active code executions',
            registry=self.registry
        )
        
        # API Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'Duration of HTTP requests in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # System Metrics
        self.system_errors_total = Counter(
            'system_errors_total',
            'Total number of system errors',
            ['error_type'],
            registry=self.registry
        )
    
    def record_llm_request(
        self,
        provider: str,
        model_id: str,
        operation: str,
        duration_seconds: float,
        token_usage: Optional[Dict[str, int]] = None,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> None:
        """Record LLM request metrics."""
        
        labels = [provider, model_id, operation]
        
        # Record request count
        self.llm_requests_total.labels(*labels).inc()
        
        # Record duration
        self.llm_request_duration.labels(*labels).observe(duration_seconds)
        
        # Record token usage
        if token_usage:
            for token_type, count in token_usage.items():
                self.llm_tokens_used.labels(provider, model_id, token_type).inc(count)
        
        # Record errors
        if not success and error_type:
            self.llm_errors_total.labels(provider, model_id, error_type).inc()
    
    def record_code_generation(self, status: str) -> None:
        """Record code generation metrics."""
        self.code_generation_total.labels(status).inc()
    
    def record_code_validation(self, result: str) -> None:
        """Record code validation metrics."""
        self.code_validation_total.labels(result).inc()
    
    def record_security_violation(self, violation_type: str) -> None:
        """Record security violation metrics."""
        self.security_violations_total.labels(violation_type).inc()
    
    def record_code_execution(
        self,
        status: str,
        duration_seconds: float,
        memory_usage_mb: Optional[float] = None
    ) -> None:
        """Record code execution metrics."""
        self.code_execution_total.labels(status).inc()
        self.code_execution_duration.observe(duration_seconds)
        
        if memory_usage_mb:
            self.docker_memory_usage.observe(memory_usage_mb)
    
    def record_job(self, status: str) -> None:
        """Record job metrics."""
        self.jobs_total.labels(status).inc()
    
    def set_queue_size(self, size: int) -> None:
        """Set current queue size."""
        self.job_queue_size.set(size)
    
    def increment_active_executions(self) -> None:
        """Increment active executions counter."""
        self.active_executions.inc()
    
    def decrement_active_executions(self) -> None:
        """Decrement active executions counter."""
        self.active_executions.dec()
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ) -> None:
        """Record HTTP request metrics."""
        self.http_requests_total.labels(method, endpoint, str(status_code)).inc()
        self.http_request_duration.labels(method, endpoint).observe(duration_seconds)
    
    def record_system_error(self, error_type: str) -> None:
        """Record system error metrics."""
        self.system_errors_total.labels(error_type).inc()
    
    def start_metrics_server(self, port: int = 8001) -> None:
        """Start Prometheus metrics server."""
        start_http_server(port, registry=self.registry)


# Global metrics instance
_metrics_instance = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get global metrics instance (singleton)."""
    global _metrics_instance
    
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MetricsCollector()
    
    return _metrics_instance


class MetricsContext:
    """Context manager for measuring operation duration."""
    
    def __init__(
        self,
        operation_name: str,
        metrics_callback: callable,
        **labels
    ):
        self.operation_name = operation_name
        self.metrics_callback = metrics_callback
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        self.metrics_callback(
            duration_seconds=duration,
            success=success,
            error_type=exc_type.__name__ if exc_type else None,
            **self.labels
        )


def measure_llm_request(provider: str, model_id: str, operation: str):
    """Decorator/context manager for measuring LLM requests."""
    metrics = get_metrics()
    
    def callback(**kwargs):
        metrics.record_llm_request(provider, model_id, operation, **kwargs)
    
    return MetricsContext(f"llm_request_{operation}", callback)


def measure_code_execution():
    """Decorator/context manager for measuring code execution."""
    metrics = get_metrics()
    
    def callback(duration_seconds, success, **kwargs):
        status = "success" if success else "failure"
        metrics.record_code_execution(status, duration_seconds)
    
    return MetricsContext("code_execution", callback) 