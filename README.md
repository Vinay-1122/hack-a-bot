# LLM Plot Generator Service

A service that generates interactive Plotly visualizations from dataframes using LLM-generated code executed in isolated Docker environments.

## Features

- **LLM-Powered**: Uses Gemini 2.0 Flash to generate intelligent plotting code
- **Isolated Execution**: Runs generated code in secure Docker containers
- **Interactive Plots**: Creates highly detailed and interactive Plotly visualizations
- **Dashboard Support**: Automatically creates dashboards for multiple dataframes
- **REST API**: Simple HTTP API for easy integration
- **Code Security**: Validates and sanitizes generated code before execution
- **Automatic Cleanup**: Cleans up temporary files and containers

## Architecture

1. **API Layer**: FastAPI service that handles requests
2. **LLM Integration**: Gemini 2.0 Flash for intelligent code generation
3. **Code Validation**: Security checks and sanitization of generated code
4. **Isolated Execution**: Docker containers for secure code execution
5. **Plot Generation**: Plotly for interactive visualizations

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Gemini API Key (from Google AI Studio)

## Setup

### Option 1: Run Locally (Recommended for Development)

This runs the service on your host machine, avoiding Docker-in-Docker complications:

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd simple_llm_plot
   ```

2. **Set your Gemini API key**
   ```bash
   # Windows
   set GEMINI_API_KEY=your_api_key_here
   
   # Linux/macOS
   export GEMINI_API_KEY=your_api_key_here
   ```

3. **Run the setup script**
   ```bash
   python run_local.py
   ```

This script will:
- Check Docker installation
- Install Python dependencies
- Build the execution Docker image
- Start the service at http://localhost:8000

### Option 2: Run with Docker Compose (Production)

For containerized deployment with Docker-in-Docker:

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd simple_llm_plot
   cp env.example .env
   # Edit .env with your Gemini API key
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

### Manual Setup Steps

If you prefer manual setup:

1. **Build the execution image**
   ```bash
   docker build -f Dockerfile.execution -t plot-executor:latest .
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Docker execution (optional)**
   ```bash
   python test_docker_simple.py
   ```

4. **Start the service**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   python main.py
   ```

## API Usage

### POST `/generate-plot`

Generate interactive plots from dataframes.

**Parameters:**
- `files`: One or more CSV files containing dataframes
- `schema`: JSON string with database schema information
- `user_question`: The original question that generated the SQL query

**Example:**
```python
import requests
import json

files = [
    ('files', ('sales.csv', open('sales.csv', 'rb'), 'text/csv')),
    ('files', ('revenue.csv', open('revenue.csv', 'rb'), 'text/csv'))
]

data = {
    'schema': json.dumps({
        "tables": {
            "sales": {"columns": {"date": "DATE", "amount": "FLOAT", "region": "VARCHAR"}},
            "revenue": {"columns": {"product": "VARCHAR", "revenue": "FLOAT"}}
        }
    }),
    'user_question': 'Show sales trends by region and revenue by product'
}

response = requests.post('http://localhost:8000/generate-plot', files=files, data=data)
result = response.json()

# Save the HTML plot
with open('plot.html', 'w') as f:
    f.write(result['html'])
```

### GET `/health`

Health check endpoint.

## Response Format

```json
{
    "html": "<html>...</html>",
    "status": "success"
}
```

The `html` field contains the complete HTML with embedded Plotly visualization.

## Features

### Code Security
- **Import Validation**: Only allows safe libraries (pandas, plotly, numpy, math, datetime)
- **Function Call Checks**: Blocks dangerous functions (eval, exec, open, etc.)
- **Pattern Matching**: Detects dangerous code patterns
- **File Operation Restrictions**: Only allows specific output operations
- **AST Analysis**: Parses and validates code structure

### Single DataFrame
- Creates a comprehensive interactive plot
- Automatically selects appropriate chart types
- Includes detailed hover information and interactivity

### Multiple DataFrames
- Automatically creates a dashboard layout
- Uses subplots to organize multiple visualizations
- Maintains consistent styling across plots

### Security
- Code execution is isolated in Docker containers
- Network access is disabled during execution
- Resource limits prevent abuse
- Automatic cleanup of temporary files

## Testing

Run the test client:
```bash
export GEMINI_API_KEY=your_api_key_here
python test_client.py
```

This will:
1. Create sample dataframes
2. Send them to the service
3. Save the generated plot as `output_plot.html`

## Troubleshooting

### Docker Connection Issues

If you see errors like "Not supported URL scheme http+docker", run the diagnostic:

```bash
python docker_check.py
```

**Common solutions:**

**Windows:**
1. Ensure Docker Desktop is running
2. Check Docker Desktop settings → General → "Expose daemon on tcp://localhost:2376 without TLS"
3. Run PowerShell as Administrator
4. Restart Docker Desktop

**Linux:**
1. Start Docker daemon: `sudo systemctl start docker`
2. Add user to docker group: `sudo usermod -aG docker $USER`
3. Log out and back in
4. Check permissions: `ls -la /var/run/docker.sock`

**macOS:**
1. Ensure Docker Desktop is running (check menu bar)
2. Restart Docker Desktop if needed

### Code Validation Errors

If you get "Code validation failed" errors:

1. Check the specific validation error message
2. The LLM might be generating code with unauthorized imports
3. Try regenerating by making the request again
4. Check logs for specific security violations

### Performance Issues

1. Increase Docker memory limit if needed
2. Check system resources
3. Monitor container execution time
4. Consider reducing dataframe size for testing

## Configuration

Environment variables:
- `GEMINI_API_KEY`: Your Gemini API key (required)

## Limitations

- Maximum execution time: 120 seconds
- Memory limit: 512MB per execution
- Network access is disabled during code execution
- Only supports CSV input format
- Only allows specific Python libraries for security

## Error Handling

The service includes comprehensive error handling:
- Invalid input validation
- Code security validation
- Code execution timeout
- Docker container failures
- LLM API errors
- Automatic cleanup on failures

## Development

For development, you can run the service locally:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_api_key_here
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Security Considerations

This service implements multiple security layers:

1. **Code Validation**: AST-based analysis prevents malicious code
2. **Import Restrictions**: Only safe libraries are allowed
3. **Container Isolation**: Network-disabled containers
4. **Resource Limits**: CPU and memory constraints
5. **Temporary Files**: Automatic cleanup of all temporary data

## License

This is a POC (Proof of Concept) service for demonstration purposes. 