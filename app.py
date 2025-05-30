import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import pandas as pd
import io
import json
import uuid

from config import settings
from core.job_manager import JobManager, JobStatus
from utils.logging_config import setup_logging, get_logger
from utils.metrics import get_metrics
from llm_providers.llm_factory import get_available_providers, get_provider_models

# Setup logging
setup_logging(settings.LOG_LEVEL, json_logs=True)
logger = get_logger(__name__)

# Global instances
job_manager = None
metrics = get_metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global job_manager
    
    # Startup
    try:
        logger.info("Starting Multi-LLM Plotting Service")
        
        # Initialize job manager
        job_manager = JobManager()
        
        # Start metrics server
        metrics.start_metrics_server(settings.PROMETHEUS_PORT)
        
        logger.info(f"Service started successfully on port {settings.API_PORT}")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-LLM Plotting Service")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Multi-LLM Plotting Service",
    description="Production-ready plotting service with multi-LLM support (Gemini & Bedrock)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)


# Pydantic models
class PlotRequest(BaseModel):
    """Request model for plot generation."""
    user_question: str = Field(..., description="User's plotting question or request")
    additional_context: Optional[str] = Field(None, description="Additional context for code generation")
    timeout_seconds: Optional[int] = Field(300, description="Execution timeout in seconds")
    priority: int = Field(0, description="Job priority (higher = more important)")
    
    @validator('user_question')
    def validate_user_question(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('User question must be at least 3 characters long')
        return v.strip()
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        if v and (v < 10 or v > 1800):  # 10 seconds to 30 minutes
            raise ValueError('Timeout must be between 10 and 1800 seconds')
        return v


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output_html: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "created_at": 1634567890.123,
                "started_at": 1634567891.456,
                "completed_at": 1634567895.789,
                "output_html": "<html>...</html>",
                "error": None,
                "metadata": {
                    "execution_time_ms": 4333,
                    "memory_usage_mb": 128.5
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    healthy: bool
    timestamp: float
    components: Dict[str, Any]


class ProvidersResponse(BaseModel):
    """Response model for available providers."""
    providers: List[str]
    models: Dict[str, List[str]]


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication (implement proper auth in production)."""
    if not credentials and settings.API_KEY_HEADER:
        raise HTTPException(status_code=401, detail="API key required")
    return {"user_id": "authenticated_user"}  # Simplified


def require_job_manager():
    """Dependency to ensure job manager is available."""
    global job_manager
    if job_manager is None:
        raise HTTPException(status_code=503, detail="Job manager not initialized")
    return job_manager


async def track_request_metrics(request: Request):
    """Middleware to track request metrics."""
    start_time = time.time()
    
    def record_metrics(response):
        duration = time.time() - start_time
        metrics.record_http_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration
        )
    
    return record_metrics


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        job_mgr = require_job_manager()
        health_status = job_mgr.health_check()
        
        return HealthResponse(**health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            healthy=False,
            timestamp=time.time(),
            components={"error": str(e)}
        )


@app.get("/health/liveness")
async def liveness_probe():
    """Simple liveness probe for Kubernetes."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/readiness")
async def readiness_probe():
    """Readiness probe for Kubernetes."""
    try:
        job_mgr = require_job_manager()
        health_status = job_mgr.health_check()
        
        if health_status["healthy"]:
            return {"status": "ready", "timestamp": time.time()}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


# Provider information endpoints
@app.get("/providers", response_model=ProvidersResponse)
async def get_providers():
    """Get available LLM providers and their supported models."""
    try:
        providers = get_available_providers()
        models = {}
        
        for provider in providers:
            try:
                models[provider] = get_provider_models(provider)
            except Exception as e:
                logger.warning(f"Failed to get models for provider {provider}: {e}")
                models[provider] = []
        
        return ProvidersResponse(providers=providers, models=models)
        
    except Exception as e:
        logger.error(f"Failed to get providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve provider information")


# Main plotting endpoints
@app.post("/plot", response_model=Dict[str, str])
async def create_plot_job(
    request: PlotRequest,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    job_mgr: JobManager = Depends(require_job_manager)
):
    """
    Create a new plotting job with uploaded data files.
    
    Supports CSV, Excel, and JSON files. Returns a job ID for tracking.
    """
    start_time = time.time()
    
    try:
        # Validate file uploads
        if not files:
            raise HTTPException(status_code=400, detail="At least one data file is required")
        
        if len(files) > 10:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
        
        # Process uploaded files
        dataframes = {}
        for file in files:
            if not file.filename:
                continue
                
            # Validate file size (e.g., 50MB limit)
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} exceeds 50MB limit"
                )
            
            # Parse file based on extension
            try:
                df = await parse_uploaded_file(file.filename, content)
                if df is not None and not df.empty:
                    # Use filename without extension as DataFrame name
                    name = file.filename.rsplit('.', 1)[0]
                    dataframes[name] = df
                    
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to parse file {file.filename}: {str(e)}"
                )
        
        if not dataframes:
            raise HTTPException(status_code=400, detail="No valid data files provided")
        
        # Validate DataFrame sizes
        total_rows = sum(len(df) for df in dataframes.values())
        if total_rows > 100000:  # 100k rows limit
            raise HTTPException(
                status_code=400,
                detail="Total dataset size exceeds 100,000 rows limit"
            )
        
        # Submit job
        job_id = job_mgr.submit_job(
            user_question=request.user_question,
            dataframes=dataframes,
            additional_context=request.additional_context,
            timeout_seconds=request.timeout_seconds,
            priority=request.priority
        )
        
        duration = time.time() - start_time
        logger.info(f"Created plot job {job_id} in {duration:.2f}s")
        
        return {
            "job_id": job_id,
            "status": "submitted",
            "message": "Job submitted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create plot job: {e}")
        metrics.record_system_error("job_creation_error")
        raise HTTPException(status_code=500, detail="Failed to create plot job")


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    job_mgr: JobManager = Depends(require_job_manager)
):
    """Get the status and result of a plotting job."""
    try:
        # Validate job ID format
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        
        result = job_mgr.get_job_status(job_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=result.job_id,
            status=result.status.value,
            created_at=result.created_at,
            started_at=result.started_at,
            completed_at=result.completed_at,
            output_html=result.output_html,
            error=result.error,
            metadata=result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job status")


@app.get("/job/{job_id}/result", response_class=HTMLResponse)
async def get_job_result(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    job_mgr: JobManager = Depends(require_job_manager)
):
    """Get the HTML result of a completed plotting job."""
    try:
        # Validate job ID format
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        
        result = job_mgr.get_job_status(job_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if result.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Job not completed. Status: {result.status.value}"
            )
        
        if not result.output_html:
            raise HTTPException(status_code=404, detail="No output available")
        
        return HTMLResponse(content=result.output_html)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job result for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job result")


@app.delete("/job/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    job_mgr: JobManager = Depends(require_job_manager)
):
    """Cancel a pending or processing job."""
    try:
        # Validate job ID format
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        
        success = job_mgr.cancel_job(job_id)
        
        if success:
            return {"message": f"Job {job_id} cancelled successfully"}
        else:
            raise HTTPException(
                status_code=400, 
                detail="Job cannot be cancelled (not found or already completed)"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")


# Utility endpoints
@app.get("/metrics")
async def get_metrics_endpoint():
    """Redirect to Prometheus metrics (for debugging)."""
    return {"message": f"Metrics available at http://localhost:{settings.PROMETHEUS_PORT}/metrics"}


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Multi-LLM Plotting Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "providers": "/providers",
            "plot": "/plot",
            "job_status": "/job/{job_id}",
            "job_result": "/job/{job_id}/result"
        },
        "supported_providers": get_available_providers()
    }


# Helper functions
async def parse_uploaded_file(filename: str, content: bytes) -> Optional[pd.DataFrame]:
    """Parse uploaded file into DataFrame."""
    try:
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'csv':
            return pd.read_csv(io.BytesIO(content))
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(io.BytesIO(content))
        elif file_extension == 'json':
            return pd.read_json(io.BytesIO(content))
        elif file_extension == 'parquet':
            return pd.read_parquet(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        logger.error(f"Failed to parse file {filename}: {e}")
        raise


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    metrics.record_system_error("unhandled_exception")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    ) 