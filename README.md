# Multi-LLM Plotting Service

A production-ready, scalable plotting service that leverages multiple Large Language Models (Google Gemini and AWS Bedrock) to generate data visualizations from natural language queries.

## Features

### 🚀 **Multi-LLM Support**
- **Google Gemini**: Fast, cost-effective code generation
- **AWS Bedrock**: Enterprise-grade with multiple model options (Claude, Titan, Llama, Cohere)
- **Provider Abstraction**: Easy switching between providers with unified interface

### 🔒 **Enterprise Security**
- **Sandboxed Execution**: Docker-based code isolation with resource limits
- **Static Analysis**: AST-based security scanning for forbidden operations
- **LLM Validation**: AI-powered code safety verification
- **Input Sanitization**: Comprehensive file and request validation

### 📊 **Advanced Plotting Capabilities**
- **Multiple Data Formats**: CSV, Excel, JSON, Parquet support
- **Rich Visualizations**: Plotly, Matplotlib, Seaborn integration
- **Interactive Dashboards**: Dash-based multi-chart visualizations
- **Export Options**: HTML, PNG, PDF output formats

### ⚡ **Production Infrastructure**
- **Asynchronous Processing**: Celery-based job queue system
- **Horizontal Scaling**: Multi-worker architecture
- **Caching Layer**: Redis for performance optimization
- **Health Monitoring**: Comprehensive health checks and metrics

### 📈 **Observability**
- **Prometheus Metrics**: Detailed performance and usage metrics
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Grafana Dashboards**: Real-time monitoring and alerting
- **Distributed Tracing**: Request lifecycle tracking

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- Google Gemini API key OR AWS credentials

### 1. Clone and Setup
```bash
git clone <repository-url>
cd llm_plotting
cp env.example .env
# Edit .env with your API keys and configuration
```

### 2. Configure Environment
```bash
# Required: Set your LLM provider credentials
export GEMINI_API_KEY="your-gemini-api-key"
# OR
export BEDROCK_ACCESS_KEY_ID="your-aws-access-key"
export BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key"
```

### 3. Start Services
```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8000/health
```

### 4. Create Your First Plot
```bash
# Upload data and create plot
curl -X POST "http://localhost:8000/plot" \
  -F "files=@data/sales_data.csv" \
  -F "user_question=Create a bar chart showing sales by region"

# Get job status
curl "http://localhost:8000/job/{job_id}"

# View result
curl "http://localhost:8000/job/{job_id}/result"
```

## API Documentation

### Endpoints

#### `POST /plot`
Create a new plotting job with uploaded data files.

**Request:**
```json
{
  "user_question": "Create a scatter plot of price vs performance",
  "additional_context": "Focus on correlation patterns",
  "timeout_seconds": 300,
  "priority": 1
}
```

**Files:** Upload CSV, Excel, JSON, or Parquet files

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "submitted",
  "message": "Job submitted successfully"
}
```

#### `GET /job/{job_id}`
Get job status and metadata.

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "created_at": 1634567890.123,
  "started_at": 1634567891.456,
  "completed_at": 1634567895.789,
  "output_html": "<html>...</html>",
  "error": null,
  "metadata": {
    "execution_time_ms": 4333,
    "memory_usage_mb": 128.5
  }
}
```

#### `GET /job/{job_id}/result`
Get the HTML visualization result.

#### `DELETE /job/{job_id}`
Cancel a pending or processing job.

#### `GET /health`
Comprehensive health check of all system components.

#### `GET /providers`
List available LLM providers and supported models.

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Celery Worker  │    │ Docker Executor │
│                 │    │                 │    │                 │
│ • REST API      │    │ • Job Processing│    │ • Code Execution│
│ • File Upload   │    │ • LLM Integration│   │ • Resource Monitoring│
│ • Authentication│    │ • Code Generation│   │ • Sandboxing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────┬───────────┼───────────┬───────────┘
                     │           │           │
                ┌─────────────────┐    ┌─────────────────┐
                │      Redis      │    │   PostgreSQL    │
                │                 │    │                 │
                │ • Job Queue     │    │ • Job Results   │
                │ • Caching       │    │ • Metadata      │
                │ • Session Store │    │ • User Data     │
                └─────────────────┘    └─────────────────┘
```

### LLM Provider Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AbstractLLMProvider                    │
├─────────────────────────────────────────────────────────┤
│ • generate_text()                                       │
│ • validate_model_id()                                   │
│ • health_check()                                        │
└─────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
     ┌─────────────────┐       ┌─────────────────┐
     │ GoogleGeminiProvider│   │ AWSBedrockProvider│
     │                 │       │                 │
     │ • gemini-pro    │       │ • Claude 3      │
     │ • gemini-pro-vision│    │ • Titan         │
     │ • gemini-1.5-pro│       │ • Llama 2       │
     └─────────────────┘       │ • Cohere        │
                               └─────────────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Primary LLM provider (gemini/bedrock) | `gemini` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `BEDROCK_REGION` | AWS Bedrock region | `us-east-1` |
| `REDIS_HOST` | Redis server host | `localhost` |
| `DOCKER_TIMEOUT` | Code execution timeout (seconds) | `300` |
| `LOG_LEVEL` | Logging level | `INFO` |

See `env.example` for complete configuration options.

### Model Configuration

#### Gemini Models
- `gemini-pro`: General purpose model
- `gemini-pro-vision`: Vision-enabled model
- `gemini-1.5-pro`: Latest high-performance model

#### Bedrock Models
- `anthropic.claude-3-sonnet-20240229-v1:0`: Balanced performance
- `anthropic.claude-3-haiku-20240307-v1:0`: Fast and cost-effective
- `amazon.titan-text-express-v1`: Amazon's foundation model
- `meta.llama2-70b-chat-v1`: Meta's Llama 2 model

## Security

### Multi-Layer Security Model

1. **Input Validation**
   - File type and size restrictions
   - Request parameter validation
   - SQL injection prevention

2. **Static Code Analysis**
   - AST parsing and security scanning
   - Forbidden import detection
   - Dangerous function identification

3. **LLM-Based Validation**
   - AI-powered code safety assessment
   - Context-aware security analysis
   - Multi-model cross-validation

4. **Sandboxed Execution**
   - Docker container isolation
   - Network restrictions
   - Resource limits (CPU, memory, I/O)
   - Read-only filesystem

5. **Runtime Monitoring**
   - Real-time resource usage tracking
   - Anomaly detection
   - Automatic termination of suspicious processes

## Monitoring

### Metrics Available

- **LLM Metrics**: Request count, latency, token usage, error rates
- **Execution Metrics**: Code execution time, memory usage, success rates
- **System Metrics**: Queue size, active workers, resource utilization
- **Security Metrics**: Violation counts, blocked requests, threat detection

### Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Celery Flower**: http://localhost:5555

## Development

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis and PostgreSQL
docker-compose up redis postgres -d

# Run the application
python app.py

# Run Celery worker (separate terminal)
celery -A core.job_manager worker --loglevel=info
```

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Load tests
pytest tests/load/
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Deployment

### Production Deployment

1. **Container Registry**
   ```bash
   docker build -t plotting-service:latest .
   docker push your-registry/plotting-service:latest
   ```

2. **Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

3. **AWS ECS/Fargate**
   ```bash
   # Use provided CloudFormation templates
   aws cloudformation deploy --template-file aws/ecs-service.yml
   ```

### Scaling Considerations

- **Horizontal Scaling**: Add more Celery workers
- **Vertical Scaling**: Increase container resources
- **Database**: Use managed PostgreSQL (RDS, Cloud SQL)
- **Cache**: Use managed Redis (ElastiCache, MemoryStore)
- **Load Balancing**: ALB/NLB for traffic distribution

## Troubleshooting

### Common Issues

**Connection Errors**
```bash
# Check service health
curl http://localhost:8000/health

# Check container logs
docker-compose logs plotting-service
```

**Performance Issues**
```bash
# Monitor resource usage
docker stats

# Check queue size
curl http://localhost:8000/metrics | grep queue_size
```

**LLM Provider Issues**
```bash
# Test provider connectivity
curl http://localhost:8000/providers

# Check API key configuration
docker-compose exec plotting-service env | grep GEMINI_API_KEY
```

### Support

- **Documentation**: Full API docs at `/docs`
- **Health Checks**: Comprehensive health monitoring at `/health`
- **Metrics**: Detailed metrics at `/metrics`
- **Logs**: Structured JSON logs with correlation IDs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

Please ensure all tests pass and code follows the project's style guidelines. 