import time
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from celery import Celery
import pandas as pd

from config import settings, get_redis_config
from core.plot_generator import PlotGenerator, GenerationResult
from core.docker_executor import DockerExecutor, ExecutionResult
from utils.prompt_builder import DataFrameSchema
from utils.logging_config import get_logger
from utils.metrics import get_metrics

logger = get_logger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class JobRequest:
    """Job request data structure."""
    job_id: str
    user_question: str
    dataframes_schemas: List[Dict[str, Any]]
    db_schema: Optional[Dict[str, Any]] = None
    additional_context: Optional[str] = None
    timeout_seconds: Optional[int] = None
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class JobResult:
    """Job result data structure."""
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output_html: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class JobManager:
    """Manages asynchronous job processing with Redis and Celery."""
    
    def __init__(self):
        self.redis_client = self._initialize_redis()
        self.celery_app = self._initialize_celery()
        self.plot_generator = PlotGenerator()
        self.docker_executor = DockerExecutor()
        self.metrics = get_metrics()
    
    def _initialize_redis(self) -> redis.Redis:
        """Initialize Redis client."""
        try:
            redis_config = get_redis_config()
            redis_client = redis.Redis(**redis_config)
            
            # Test connection
            redis_client.ping()
            logger.info("Redis client initialized successfully")
            
            return redis_client
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    def _initialize_celery(self) -> Celery:
        """Initialize Celery app."""
        try:
            celery_app = Celery(
                'plotting_service',
                broker=settings.CELERY_BROKER_URL,
                backend=settings.CELERY_RESULT_BACKEND
            )
            
            # Configure Celery
            celery_app.conf.update(
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,
                task_routes={
                    'plotting_service.process_plotting_job': {'queue': 'plotting'},
                },
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_max_tasks_per_child=100,
            )
            
            # Register tasks
            self._register_celery_tasks(celery_app)
            
            logger.info("Celery app initialized successfully")
            return celery_app
            
        except Exception as e:
            logger.error(f"Failed to initialize Celery app: {e}")
            raise
    
    def _register_celery_tasks(self, celery_app: Celery) -> None:
        """Register Celery tasks."""
        
        @celery_app.task(name='plotting_service.process_plotting_job', bind=True)
        def process_plotting_job(self_task, job_request_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Process a plotting job."""
            job_request = JobRequest(**job_request_dict)
            return self._process_job(job_request, self_task.request.id)
        
        self.process_plotting_job = process_plotting_job
    
    def submit_job(
        self,
        user_question: str,
        dataframes: Dict[str, pd.DataFrame],
        db_schema: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        priority: int = 0
    ) -> str:
        """
        Submit a new plotting job.
        
        Args:
            user_question: User's plotting request
            dataframes: Dictionary of DataFrames
            db_schema: Optional database schema
            additional_context: Additional context for generation
            timeout_seconds: Job timeout
            priority: Job priority (higher = more important)
            
        Returns:
            str: Job ID
        """
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        try:
            # Create DataFrame schemas
            dataframes_schemas = []
            for name, df in dataframes.items():
                schema = DataFrameSchema(
                    name=name,
                    columns=df.columns.tolist(),
                    dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                    shape=df.shape,
                    sample_data=df.head(3).to_dict('records') if len(df) > 0 else None
                )
                dataframes_schemas.append(asdict(schema))
                
                # Store DataFrame in Redis with expiration
                df_key = f"dataframe:{job_id}:{name}"
                df_json = df.to_json(orient='records')
                self.redis_client.setex(df_key, 3600, df_json)  # 1 hour expiration
            
            # Create job request
            job_request = JobRequest(
                job_id=job_id,
                user_question=user_question,
                dataframes_schemas=dataframes_schemas,
                db_schema=db_schema,
                additional_context=additional_context,
                timeout_seconds=timeout_seconds,
                priority=priority
            )
            
            # Store job status
            self._update_job_status(job_id, JobStatus.PENDING, job_request=job_request)
            
            # Submit to Celery queue
            task = self.process_plotting_job.apply_async(
                args=[asdict(job_request)],
                priority=priority,
                countdown=0
            )
            
            # Store task ID for job
            self.redis_client.setex(f"task:{job_id}", 3600, task.id)
            
            logger.info(f"Submitted job {job_id} to queue")
            self.metrics.record_job("submitted")
            self.metrics.set_queue_size(self._get_queue_size())
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            self._update_job_status(job_id, JobStatus.FAILED, error=str(e))
            self.metrics.record_job("submit_failed")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get job status and result."""
        try:
            result_key = f"job_result:{job_id}"
            result_data = self.redis_client.get(result_key)
            
            if result_data:
                result_dict = json.loads(result_data)
                result_dict['status'] = JobStatus(result_dict['status'])
                return JobResult(**result_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job."""
        try:
            # Get current status
            current_result = self.get_job_status(job_id)
            if not current_result:
                return False
            
            if current_result.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False  # Already finished
            
            # Cancel Celery task
            task_id = self.redis_client.get(f"task:{job_id}")
            if task_id:
                self.celery_app.control.revoke(task_id.decode(), terminate=True)
            
            # Update status
            self._update_job_status(job_id, JobStatus.CANCELLED)
            
            logger.info(f"Cancelled job {job_id}")
            self.metrics.record_job("cancelled")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def _process_job(self, job_request: JobRequest, task_id: str) -> Dict[str, Any]:
        """Process a plotting job (called by Celery worker)."""
        job_id = job_request.job_id
        
        logger.info(f"Processing job {job_id}")
        self._update_job_status(job_id, JobStatus.PROCESSING)
        self.metrics.record_job("processing")
        
        try:
            # Load DataFrames from Redis
            dataframes = {}
            for schema_dict in job_request.dataframes_schemas:
                name = schema_dict['name']
                df_key = f"dataframe:{job_id}:{name}"
                df_json = self.redis_client.get(df_key)
                
                if df_json:
                    dataframes[name] = pd.read_json(df_json, orient='records')
                else:
                    raise ValueError(f"DataFrame {name} not found in Redis")
            
            # Convert schemas back to DataFrameSchema objects
            dataframe_schemas = [DataFrameSchema(**schema) for schema in job_request.dataframes_schemas]
            
            # Generate and validate code
            generation_result = self.plot_generator.generate_and_validate_code(
                user_question=job_request.user_question,
                dataframes=dataframe_schemas,
                db_schema=job_request.db_schema,
                job_id=job_id,
                additional_context=job_request.additional_context
            )
            
            if not generation_result.success:
                self._update_job_status(
                    job_id, 
                    JobStatus.FAILED, 
                    error=generation_result.error,
                    metadata=generation_result.generation_metadata
                )
                self.metrics.record_job("generation_failed")
                return {"status": "failed", "error": generation_result.error}
            
            # Execute code
            execution_result = self.docker_executor.execute_code(
                code=generation_result.code,
                dataframes=dataframes,
                job_id=job_id,
                timeout_seconds=job_request.timeout_seconds
            )
            
            if execution_result.success:
                # Job completed successfully
                self._update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    output_html=execution_result.output_html,
                    metadata={
                        "generation": generation_result.generation_metadata,
                        "validation": generation_result.validation_metadata,
                        "execution": execution_result.metadata,
                        "execution_time_ms": execution_result.execution_time_ms,
                        "memory_usage_mb": execution_result.memory_usage_mb
                    }
                )
                
                logger.info(f"Job {job_id} completed successfully")
                self.metrics.record_job("completed")
                
                return {
                    "status": "completed",
                    "output_html": execution_result.output_html,
                    "metadata": {
                        "execution_time_ms": execution_result.execution_time_ms,
                        "memory_usage_mb": execution_result.memory_usage_mb
                    }
                }
            else:
                # Execution failed
                self._update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error=execution_result.error,
                    metadata={
                        "generation": generation_result.generation_metadata,
                        "validation": generation_result.validation_metadata,
                        "execution": execution_result.metadata
                    }
                )
                
                logger.error(f"Job {job_id} execution failed: {execution_result.error}")
                self.metrics.record_job("execution_failed")
                
                return {"status": "failed", "error": execution_result.error}
                
        except Exception as e:
            error_msg = f"Job processing error: {str(e)}"
            logger.error(f"Job {job_id} processing error: {error_msg}")
            
            self._update_job_status(job_id, JobStatus.FAILED, error=error_msg)
            self.metrics.record_job("error")
            
            return {"status": "failed", "error": error_msg}
        
        finally:
            # Clean up DataFrames from Redis
            self._cleanup_job_data(job_id, job_request.dataframes_schemas)
    
    def _update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        job_request: Optional[JobRequest] = None,
        output_html: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update job status in Redis."""
        try:
            current_time = time.time()
            
            # Get existing result or create new one
            existing_result = self.get_job_status(job_id)
            
            if existing_result:
                result = existing_result
                result.status = status
            else:
                result = JobResult(
                    job_id=job_id,
                    status=status,
                    created_at=job_request.created_at if job_request else current_time
                )
            
            # Update timing
            if status == JobStatus.PROCESSING and not result.started_at:
                result.started_at = current_time
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if not result.completed_at:
                    result.completed_at = current_time
            
            # Update data
            if output_html:
                result.output_html = output_html
            if error:
                result.error = error
            if metadata:
                result.metadata = metadata
            
            # Store in Redis with expiration
            result_key = f"job_result:{job_id}"
            result_data = asdict(result)
            result_data['status'] = result.status.value  # Convert enum to string
            
            expiration_hours = settings.JOB_RETENTION_HOURS
            self.redis_client.setex(
                result_key,
                expiration_hours * 3600,
                json.dumps(result_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to update job status for {job_id}: {e}")
    
    def _cleanup_job_data(self, job_id: str, dataframes_schemas: List[Dict[str, Any]]) -> None:
        """Clean up job data from Redis."""
        try:
            # Remove DataFrames
            for schema in dataframes_schemas:
                name = schema['name']
                df_key = f"dataframe:{job_id}:{name}"
                self.redis_client.delete(df_key)
            
            # Remove task ID
            self.redis_client.delete(f"task:{job_id}")
            
            logger.debug(f"Cleaned up job data for {job_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup job data for {job_id}: {e}")
    
    def _get_queue_size(self) -> int:
        """Get current queue size."""
        try:
            # This is a simplified implementation
            # In production, you might want to use Celery's inspect API
            return 0  # Placeholder
        except Exception:
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on job manager."""
        health_status = {
            "healthy": True,
            "components": {},
            "timestamp": time.time()
        }
        
        # Check Redis
        try:
            self.redis_client.ping()
            health_status["components"]["redis"] = {"healthy": True}
        except Exception as e:
            health_status["components"]["redis"] = {"healthy": False, "error": str(e)}
            health_status["healthy"] = False
        
        # Check Celery (simplified)
        try:
            # In a real implementation, you'd check worker availability
            health_status["components"]["celery"] = {"healthy": True}
        except Exception as e:
            health_status["components"]["celery"] = {"healthy": False, "error": str(e)}
            health_status["healthy"] = False
        
        # Check plot generator
        plot_gen_health = self.plot_generator.health_check()
        health_status["components"]["plot_generator"] = plot_gen_health
        if not plot_gen_health["healthy"]:
            health_status["healthy"] = False
        
        # Check docker executor
        docker_health = self.docker_executor.health_check()
        health_status["components"]["docker_executor"] = docker_health
        if not docker_health["healthy"]:
            health_status["healthy"] = False
        
        return health_status 