#!/usr/bin/env python3
"""
Docker connectivity diagnostic script
"""
import docker
import platform
import subprocess
import sys

def check_docker_installation():
    """Check if Docker is installed and accessible"""
    print("Checking Docker installation...")
    
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Docker command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker command not found")
        return False

def check_docker_daemon():
    """Check if Docker daemon is running"""
    print("\nChecking Docker daemon...")
    
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
            return True
        else:
            print(f"❌ Docker daemon not accessible: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker daemon: {e}")
        return False

def test_docker_connections():
    """Test different Docker connection methods"""
    print("\nTesting Docker API connections...")
    
    connection_methods = []
    
    # Default connection
    try:
        client = docker.from_env()
        client.ping()
        print("✅ Default connection (docker.from_env()) works")
        connection_methods.append("default")
    except Exception as e:
        print(f"❌ Default connection failed: {e}")
    
    # Windows named pipe
    if platform.system() == "Windows":
        try:
            client = docker.DockerClient(base_url='npipe:////./pipe/docker_engine')
            client.ping()
            print("✅ Windows named pipe connection works")
            connection_methods.append("windows_pipe")
        except Exception as e:
            print(f"❌ Windows named pipe failed: {e}")
    
    # Unix socket
    try:
        client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        client.ping()
        print("✅ Unix socket connection works")
        connection_methods.append("unix_socket")
    except Exception as e:
        print(f"❌ Unix socket failed: {e}")
    
    # TCP connection
    try:
        client = docker.DockerClient(base_url='tcp://localhost:2376')
        client.ping()
        print("✅ TCP connection works")
        connection_methods.append("tcp")
    except Exception as e:
        print(f"❌ TCP connection failed: {e}")
    
    return connection_methods

def test_container_run():
    """Test running a simple container"""
    print("\nTesting container execution...")
    
    try:
        client = docker.from_env()
        
        # Test running a simple container
        container = client.containers.run(
            "python:3.9-slim",
            command="python -c 'print(\"Hello from Docker!\")'",
            remove=True,
            stdout=True,
            stderr=True
        )
        
        print(f"✅ Container execution successful: {container.decode().strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Container execution failed: {e}")
        return False

def show_recommendations():
    """Show recommendations based on the system"""
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    print("="*50)
    
    system = platform.system()
    
    if system == "Windows":
        print("For Windows:")
        print("1. Ensure Docker Desktop is installed and running")
        print("2. Check that Docker Desktop is set to use Windows containers or Linux containers as needed")
        print("3. Verify Docker Desktop settings allow API access")
        print("4. Try running PowerShell/Command Prompt as Administrator")
        
    elif system == "Linux":
        print("For Linux:")
        print("1. Ensure Docker daemon is running: sudo systemctl status docker")
        print("2. Start Docker if not running: sudo systemctl start docker")
        print("3. Add your user to docker group: sudo usermod -aG docker $USER")
        print("4. Log out and log back in after adding to docker group")
        print("5. Check socket permissions: ls -la /var/run/docker.sock")
        
    elif system == "Darwin":  # macOS
        print("For macOS:")
        print("1. Ensure Docker Desktop is installed and running")
        print("2. Check Docker Desktop is running in menu bar")
        print("3. Verify Docker Desktop settings allow API access")
    
    print("\nGeneral troubleshooting:")
    print("1. Restart Docker service/application")
    print("2. Check firewall settings")
    print("3. Verify no other applications are using Docker ports")

def main():
    print("Docker Connectivity Diagnostic")
    print("="*50)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Run checks
    docker_installed = check_docker_installation()
    daemon_running = check_docker_daemon()
    
    if docker_installed and daemon_running:
        working_connections = test_docker_connections()
        
        if working_connections:
            print(f"\n✅ Found {len(working_connections)} working connection(s): {', '.join(working_connections)}")
            test_container_run()
        else:
            print("\n❌ No working Docker connections found")
    
    show_recommendations()

if __name__ == "__main__":
    main() 