import os
import tempfile
import shutil
import time
import json
import subprocess
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import docker
from docker.errors import DockerException, ContainerError, ImageNotFound
import pandas as pd

from config import settings
from utils.logging_config import log_code_execution
from utils.metrics import get_metrics, measure_code_execution

logger = __import__('logging').getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output_html: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    memory_usage_mb: float = 0
    logs: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    network_io: Dict[str, int]
    block_io: Dict[str, int]


class DockerExecutor:
    """Executes Python code in sandboxed Docker containers with monitoring."""
    
    def __init__(self):
        self.docker_client = None
        self.metrics = get_metrics()
        self._initialize_docker()
    
    def _initialize_docker(self) -> None:
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def execute_code(
        self,
        code: str,
        dataframes: Dict[str, pd.DataFrame],
        job_id: str,
        timeout_seconds: int = None
    ) -> ExecutionResult:
        """
        Execute Python code in a sandboxed Docker container.
        
        Args:
            code: Python code to execute
            dataframes: Dictionary of DataFrames to make available
            job_id: Job identifier for logging
            timeout_seconds: Execution timeout (uses config default if None)
            
        Returns:
            ExecutionResult: Results of code execution
        """
        timeout_seconds = timeout_seconds or settings.DOCKER_TIMEOUT
        temp_dir = None
        container = None
        
        start_time = time.time()
        
        try:
            # Create temporary directory for execution
            temp_dir = self._create_temp_directory(job_id)
            
            # Prepare data files
            self._write_dataframes(temp_dir, dataframes)
            
            # Write code to file
            code_file = os.path.join(temp_dir, "plot_generator.py")
            self._write_code_file(code_file, code, dataframes)
            
            # Execute in Docker container
            with measure_code_execution():
                container = self._create_container(temp_dir, job_id)
                execution_result = self._execute_container(
                    container, job_id, timeout_seconds
                )
            
            # Read output
            if execution_result.success:
                output_html = self._read_output_html(temp_dir)
                execution_result.output_html = output_html
            
            execution_time_ms = (time.time() - start_time) * 1000
            execution_result.execution_time_ms = execution_time_ms
            
            # Log execution
            log_code_execution(
                logger=logger,
                job_id=job_id,
                code_length=len(code),
                execution_time_ms=execution_time_ms,
                memory_usage_mb=execution_result.memory_usage_mb,
                success=execution_result.success,
                error=execution_result.error,
                output_size_bytes=len(execution_result.output_html) if execution_result.output_html else 0
            )
            
            return execution_result
            
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            logger.error(f"Execution error for job {job_id}: {error_msg}")
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        finally:
            # Cleanup
            if container:
                self._cleanup_container(container, job_id)
            
            if temp_dir and settings.CLEANUP_TEMP_FILES:
                self._cleanup_temp_directory(temp_dir)
    
    def _create_temp_directory(self, job_id: str) -> str:
        """Create temporary directory for execution."""
        base_dir = settings.TEMP_DIR_BASE
        os.makedirs(base_dir, exist_ok=True)
        
        temp_dir = os.path.join(base_dir, f"job_{job_id}_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create data subdirectory
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        return temp_dir
    
    def _write_dataframes(self, temp_dir: str, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Write DataFrames to CSV files."""
        data_dir = os.path.join(temp_dir, "data")
        
        for name, df in dataframes.items():
            csv_path = os.path.join(data_dir, f"{name}.csv")
            df.to_csv(csv_path, index=False)
            logger.debug(f"Wrote DataFrame {name} to {csv_path}")
    
    def _write_code_file(self, code_file: str, code: str, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Write executable Python code file."""
        
        # Prepare DataFrame loading code
        loading_code = ""
        for name in dataframes.keys():
            loading_code += f"{name} = pd.read_csv('/app/data/{name}.csv')\n"
        
        # Complete code with imports and DataFrame loading
        complete_code = f"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# Load DataFrames
{loading_code}

# User code
{code}
"""
        
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(complete_code)
        
        logger.debug(f"Wrote code file to {code_file}")
    
    def _create_container(self, temp_dir: str, job_id: str) -> docker.models.containers.Container:
        """Create Docker container for code execution."""
        try:
            # Container configuration
            container_config = {
                'image': settings.DOCKER_IMAGE,
                'command': ['python', '/app/plot_generator.py'],
                'volumes': {
                    temp_dir: {'bind': '/app', 'mode': 'rw'}
                },
                'working_dir': '/app',
                'network_mode': 'none',  # No network access
                'mem_limit': settings.DOCKER_MEMORY_LIMIT,
                'cpu_quota': int(float(settings.DOCKER_CPU_LIMIT) * 100000),
                'cpu_period': 100000,
                'detach': True,
                'remove': False,  # We'll remove manually after getting stats
                'name': f"plotting_job_{job_id}",
                'environment': {
                    'PYTHONUNBUFFERED': '1',
                    'MPLBACKEND': 'Agg'  # Matplotlib backend for headless operation
                },
                'security_opt': ['no-new-privileges:true'],
                'read_only': False,  # Need write access for output
                'tmpfs': {'/tmp': 'noexec,nosuid,size=100m'}
            }
            
            container = self.docker_client.containers.create(**container_config)
            logger.info(f"Created container {container.id} for job {job_id}")
            
            return container
            
        except ImageNotFound:
            logger.error(f"Docker image {settings.DOCKER_IMAGE} not found")
            raise
        except DockerException as e:
            logger.error(f"Failed to create container for job {job_id}: {e}")
            raise
    
    def _execute_container(
        self,
        container: docker.models.containers.Container,
        job_id: str,
        timeout_seconds: int
    ) -> ExecutionResult:
        """Execute container with monitoring."""
        
        resource_monitor = None
        resource_stats = []
        
        try:
            # Start resource monitoring
            resource_monitor = threading.Thread(
                target=self._monitor_resources,
                args=(container, resource_stats),
                daemon=True
            )
            
            # Start container
            self.metrics.increment_active_executions()
            container.start()
            resource_monitor.start()
            
            logger.info(f"Started container execution for job {job_id}")
            
            # Wait for completion or timeout
            try:
                exit_code = container.wait(timeout=timeout_seconds)['StatusCode']
            except Exception as e:
                logger.warning(f"Container wait timeout for job {job_id}: {e}")
                container.kill()
                return ExecutionResult(
                    success=False,
                    error=f"Execution timeout after {timeout_seconds} seconds"
                )
            
            # Get logs
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            # Calculate resource usage
            max_memory_mb = 0
            if resource_stats:
                max_memory_mb = max(stat.memory_usage_mb for stat in resource_stats)
            
            if exit_code == 0:
                logger.info(f"Container execution successful for job {job_id}")
                return ExecutionResult(
                    success=True,
                    logs=logs,
                    memory_usage_mb=max_memory_mb,
                    metadata={
                        'exit_code': exit_code,
                        'resource_stats': len(resource_stats)
                    }
                )
            else:
                error_msg = f"Container execution failed with exit code {exit_code}"
                logger.error(f"Container execution failed for job {job_id}: {error_msg}")
                return ExecutionResult(
                    success=False,
                    error=error_msg,
                    logs=logs,
                    memory_usage_mb=max_memory_mb,
                    metadata={
                        'exit_code': exit_code,
                        'resource_stats': len(resource_stats)
                    }
                )
                
        except Exception as e:
            logger.error(f"Container execution error for job {job_id}: {e}")
            return ExecutionResult(
                success=False,
                error=f"Container execution error: {str(e)}"
            )
            
        finally:
            self.metrics.decrement_active_executions()
    
    def _monitor_resources(
        self,
        container: docker.models.containers.Container,
        resource_stats: List[ResourceUsage]
    ) -> None:
        """Monitor container resource usage."""
        try:
            while True:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    cpu_percent = 0
                    if 'cpu_stats' in stats and 'precpu_stats' in stats:
                        cpu_stats = stats['cpu_stats']
                        precpu_stats = stats['precpu_stats']
                        
                        cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
                        system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
                        
                        if system_delta > 0:
                            cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100
                    
                    # Calculate memory usage
                    memory_usage_mb = 0
                    memory_limit_mb = 0
                    if 'memory_stats' in stats:
                        memory_stats = stats['memory_stats']
                        memory_usage_mb = memory_stats.get('usage', 0) / (1024 * 1024)
                        memory_limit_mb = memory_stats.get('limit', 0) / (1024 * 1024)
                    
                    # Network and block I/O
                    network_io = stats.get('networks', {})
                    block_io = stats.get('blkio_stats', {})
                    
                    resource_usage = ResourceUsage(
                        cpu_percent=cpu_percent,
                        memory_usage_mb=memory_usage_mb,
                        memory_limit_mb=memory_limit_mb,
                        network_io=network_io,
                        block_io=block_io
                    )
                    
                    resource_stats.append(resource_usage)
                    
                    time.sleep(1)  # Monitor every second
                    
                except Exception as e:
                    logger.debug(f"Resource monitoring error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Resource monitor thread error: {e}")
    
    def _read_output_html(self, temp_dir: str) -> Optional[str]:
        """Read output HTML file."""
        output_path = os.path.join(temp_dir, "data", "output.html")
        
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Read output HTML ({len(content)} bytes)")
                return content
            except Exception as e:
                logger.error(f"Failed to read output HTML: {e}")
                return None
        else:
            logger.warning("Output HTML file not found")
            return None
    
    def _cleanup_container(self, container: docker.models.containers.Container, job_id: str) -> None:
        """Clean up Docker container."""
        try:
            container.remove(force=True)
            logger.debug(f"Removed container for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to remove container for job {job_id}: {e}")
    
    def _cleanup_temp_directory(self, temp_dir: str) -> None:
        """Clean up temporary directory."""
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Removed temporary directory {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Docker executor."""
        health_status = {
            "healthy": True,
            "docker_available": False,
            "image_available": False,
            "timestamp": time.time()
        }
        
        try:
            # Check Docker daemon
            self.docker_client.ping()
            health_status["docker_available"] = True
            
            # Check if required image is available
            try:
                self.docker_client.images.get(settings.DOCKER_IMAGE)
                health_status["image_available"] = True
            except ImageNotFound:
                health_status["healthy"] = False
                health_status["image_available"] = False
                
        except Exception as e:
            health_status["healthy"] = False
            health_status["error"] = str(e)
        
        return health_status 