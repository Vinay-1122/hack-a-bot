import tempfile
import os
import shutil
import uuid
import pandas as pd
from typing import List, Dict
import asyncio
import time
import platform
import subprocess
import json

class CodeExecutor:
    def __init__(self):
        self.working_dir = "/tmp/plot_execution"
        self.docker_cmd = self._find_docker_executable()
        
    def _find_docker_executable(self):
        """Find Docker executable path with comprehensive search"""
        
        # Cache file to store found Docker path
        cache_file = os.path.join(os.path.dirname(__file__), '.docker_path_cache')
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_path = f.read().strip()
                    if self._test_docker_path(cached_path):
                        return cached_path
            except:
                pass
        
        print("Searching for Docker executable...")
        
        # Method 1: Try simple 'docker' command
        if self._test_docker_path('docker'):
            self._cache_docker_path('docker', cache_file)
            return 'docker'
        
        # Method 2: On Windows, try common Docker Desktop paths
        if platform.system() == "Windows":
            possible_paths = [
                'C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe',
                'C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker',
                'docker.exe'
            ]
            
            for path in possible_paths:
                if self._test_docker_path(path):
                    self._cache_docker_path(path, cache_file)
                    return path
            
            # Try using 'where' command to find docker
            try:
                result = subprocess.run(['where', 'docker'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    docker_path = result.stdout.strip().split('\n')[0]
                    if self._test_docker_path(docker_path):
                        self._cache_docker_path(docker_path, cache_file)
                        return docker_path
            except:
                pass
            
            # Try searching in common PATH locations
            path_dirs = os.environ.get('PATH', '').split(os.pathsep)
            for path_dir in path_dirs:
                docker_exe = os.path.join(path_dir, 'docker.exe')
                if os.path.exists(docker_exe) and self._test_docker_path(docker_exe):
                    self._cache_docker_path(docker_exe, cache_file)
                    return docker_exe
                
                docker_no_ext = os.path.join(path_dir, 'docker')
                if os.path.exists(docker_no_ext) and self._test_docker_path(docker_no_ext):
                    self._cache_docker_path(docker_no_ext, cache_file)
                    return docker_no_ext
        
        # Method 3: On Unix-like systems, try 'which'
        else:
            try:
                result = subprocess.run(['which', 'docker'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    docker_path = result.stdout.strip()
                    if self._test_docker_path(docker_path):
                        self._cache_docker_path(docker_path, cache_file)
                        return docker_path
            except:
                pass
        
        # Method 4: Search in standard locations
        standard_locations = [
            '/usr/bin/docker',
            '/usr/local/bin/docker',
            '/opt/docker/bin/docker'
        ]
        
        for location in standard_locations:
            if self._test_docker_path(location):
                self._cache_docker_path(location, cache_file)
                return location
        
        # If all methods fail, provide detailed error
        error_msg = f"""
Docker executable not found. Searched:
- Simple 'docker' command
- Windows Docker Desktop paths
- PATH environment variable
- Standard system locations

Current PATH: {os.environ.get('PATH', 'Not found')}
Platform: {platform.system()}

Please ensure Docker is installed and accessible.
"""
        
        raise Exception(error_msg.strip())
    
    def _test_docker_path(self, docker_path):
        """Test if a Docker path is valid and working"""
        try:
            result = subprocess.run([docker_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return False
    
    def _cache_docker_path(self, docker_path, cache_file):
        """Cache the found Docker path for future use"""
        try:
            with open(cache_file, 'w') as f:
                f.write(docker_path)
            print(f"Docker executable found and cached: {docker_path}")
        except:
            print(f"Docker executable found: {docker_path} (caching failed)")
        
    async def execute_code(self, code: str, dataframes: List[Dict]) -> str:
        """
        Execute Python code in isolated Docker environment
        
        Args:
            code: Python code to execute
            dataframes: List of dataframe information
            
        Returns:
            HTML content of generated plot
        """
        
        execution_id = str(uuid.uuid4())
        temp_dir = f"/tmp/plot_exec_{execution_id}"
        
        # For Windows, use a different temp directory
        if platform.system() == "Windows":
            temp_dir = os.path.join(os.environ.get('TEMP', 'C:\\temp'), f"plot_exec_{execution_id}")
        
        try:
            # Create temporary directory
            os.makedirs(temp_dir, exist_ok=True)
            
            # Prepare data files
            self._prepare_data_files(dataframes, temp_dir)
            
            # Prepare execution script
            script_path = self._prepare_execution_script(code, dataframes, temp_dir)
            
            # Run in Docker container using subprocess
            html_content = await self._run_in_docker_subprocess(temp_dir, script_path)
            
            return html_content
            
        finally:
            # Cleanup
            self._cleanup(temp_dir)
    
    def _prepare_data_files(self, dataframes: List[Dict], temp_dir: str):
        """Save dataframes as CSV files in temp directory"""
        for i, df_info in enumerate(dataframes):
            csv_path = os.path.join(temp_dir, f"df{i+1}.csv")
            df_info['data'].to_csv(csv_path, index=False)
    
    def _prepare_execution_script(self, code: str, dataframes: List[Dict], temp_dir: str) -> str:
        """Prepare the execution script with data loading"""
        
        # Generate data loading code
        data_loading_code = "import pandas as pd\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\nimport plotly.io as pio\n\n"
        
        for i, df_info in enumerate(dataframes):
            data_loading_code += f"df{i+1} = pd.read_csv('/tmp/df{i+1}.csv')\n"
        
        data_loading_code += "\n"
        
        # Combine with generated code
        full_script = data_loading_code + code
        
        # Write script file
        script_path = os.path.join(temp_dir, "plot_script.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(full_script)
        
        return script_path
    
    async def _run_in_docker_subprocess(self, temp_dir: str, script_path: str) -> str:
        """Run the script in Docker container using subprocess"""
        
        try:
            # Prepare volume mapping for different platforms
            if platform.system() == "Windows":
                # Convert Windows path to Docker-compatible format for volume mounting
                # Convert C:\Users\... to /c/Users/...
                docker_temp_dir = temp_dir.replace('\\', '/').replace('C:', '/c')
                volume_mount = f"{docker_temp_dir}:/tmp"
            else:
                volume_mount = f"{temp_dir}:/tmp"
            
            # Build Docker command
            docker_cmd = [
                self.docker_cmd, 'run',
                '--rm',  # Remove container after execution
                '--network=none',  # Disable network
                '--memory=512m',  # Memory limit
                '--cpus=0.5',  # CPU limit
                '-v', volume_mount,  # Volume mount
                '-w', '/tmp',  # Working directory
                'plot-executor:latest',  # Use custom image with packages
                'python', '/tmp/plot_script.py'  # Direct execution
            ]
            
            print(f"Running Docker command: {' '.join(docker_cmd)}")
            
            # Run the Docker container
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                # Wait for completion with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=120.0  # 2 minute timeout
                )
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    raise Exception(f"Docker execution failed (exit code {process.returncode}): {error_msg}")
                
                print(f"Docker stdout: {stdout.decode()}")
                if stderr:
                    print(f"Docker stderr: {stderr.decode()}")
                
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                raise Exception("Docker execution timed out")
            
            # Read generated HTML
            html_path = os.path.join(temp_dir, "output.html")
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise Exception(f"No output HTML file generated. Check logs above.")
                
        except Exception as e:
            raise Exception(f"Docker subprocess execution error: {str(e)}")
    
    def _cleanup(self, temp_dir: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Cleanup warning: {e}")  # Log but don't fail 