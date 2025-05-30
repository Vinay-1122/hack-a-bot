# Multi-LLM Plotting Service Makefile

# Variables
DOCKER_IMAGE_NAME = plotting-service
DOCKER_TAG = latest
COMPOSE_FILE = docker-compose.yml

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Multi-LLM Plotting Service Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# Development Commands
.PHONY: install
install: ## Install Python dependencies
	pip install -r requirements.txt

.PHONY: dev-setup
dev-setup: ## Setup development environment
	cp env.example .env
	docker-compose up -d redis postgres
	@echo "Edit .env file with your API keys before starting the service"

.PHONY: dev
dev: ## Run development server
	python app.py

.PHONY: worker
worker: ## Run Celery worker
	python worker.py

.PHONY: flower
flower: ## Run Celery Flower monitoring
	celery -A core.job_manager flower --port=5555

# Docker Commands
.PHONY: build
build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

.PHONY: up
up: ## Start all services with Docker Compose
	docker-compose up -d

.PHONY: down
down: ## Stop all services
	docker-compose down

.PHONY: restart
restart: down up ## Restart all services

.PHONY: logs
logs: ## Show logs from all services
	docker-compose logs -f

.PHONY: logs-api
logs-api: ## Show logs from API service only
	docker-compose logs -f plotting-service

.PHONY: logs-worker
logs-worker: ## Show logs from worker service only
	docker-compose logs -f celery-worker

# Testing Commands
.PHONY: test
test: ## Run unit tests
	pytest tests/ -v

.PHONY: test-integration
test-integration: ## Run integration tests
	pytest tests/integration/ -v

.PHONY: test-all
test-all: ## Run all tests
	pytest tests/ -v --cov=.

.PHONY: test-load
test-load: ## Run load tests
	pytest tests/load/ -v

# Code Quality Commands
.PHONY: format
format: ## Format code with black
	black .

.PHONY: lint
lint: ## Lint code with flake8
	flake8 .

.PHONY: type-check
type-check: ## Run type checking with mypy
	mypy .

.PHONY: quality
quality: format lint type-check ## Run all code quality checks

# Health Check Commands
.PHONY: health
health: ## Check service health
	curl -f http://localhost:8000/health || exit 1

.PHONY: health-full
health-full: ## Check health of all components
	@echo "Checking API health..."
	curl -f http://localhost:8000/health
	@echo "\nChecking Prometheus..."
	curl -f http://localhost:9090/-/healthy
	@echo "\nChecking Grafana..."
	curl -f http://localhost:3000/api/health
	@echo "\nAll services healthy!"

# Database Commands
.PHONY: db-migrate
db-migrate: ## Run database migrations
	alembic upgrade head

.PHONY: db-reset
db-reset: ## Reset database (WARNING: destroys data)
	docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS plotting_service;"
	docker-compose exec postgres psql -U postgres -c "CREATE DATABASE plotting_service;"
	$(MAKE) db-migrate

# Monitoring Commands
.PHONY: metrics
metrics: ## Show Prometheus metrics
	curl http://localhost:8001/metrics

.PHONY: grafana
grafana: ## Open Grafana dashboard
	@echo "Opening Grafana at http://localhost:3000 (admin/admin)"
	@which open > /dev/null && open http://localhost:3000 || echo "Navigate to http://localhost:3000"

.PHONY: flower-ui
flower-ui: ## Open Celery Flower UI
	@echo "Opening Flower at http://localhost:5555"
	@which open > /dev/null && open http://localhost:5555 || echo "Navigate to http://localhost:5555"

# Cleanup Commands
.PHONY: clean
clean: ## Clean up Docker resources
	docker-compose down -v
	docker system prune -f

.PHONY: clean-all
clean-all: ## Clean up everything (WARNING: removes all data)
	docker-compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# Deployment Commands
.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

.PHONY: deploy-prod
deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	@echo "Make sure you have set the production environment variables!"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# API Testing Commands
.PHONY: test-api
test-api: ## Test API endpoints
	@echo "Testing API endpoints..."
	curl -X GET http://localhost:8000/
	curl -X GET http://localhost:8000/health
	curl -X GET http://localhost:8000/providers

.PHONY: sample-request
sample-request: ## Make a sample plotting request
	@echo "Making sample plotting request..."
	@echo "Note: You need to have a CSV file at ./sample_data.csv"
	curl -X POST "http://localhost:8000/plot" \
		-F "files=@sample_data.csv" \
		-F "user_question=Create a bar chart showing the distribution of data"

# Documentation Commands
.PHONY: docs
docs: ## Generate documentation
	@echo "API documentation available at:"
	@echo "  Swagger UI: http://localhost:8000/docs"
	@echo "  ReDoc: http://localhost:8000/redoc"

.PHONY: docs-open
docs-open: ## Open API documentation
	@which open > /dev/null && open http://localhost:8000/docs || echo "Navigate to http://localhost:8000/docs"

# Security Commands
.PHONY: security-check
security-check: ## Run security checks
	bandit -r . -x tests/
	safety check

# Performance Commands
.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	ab -n 100 -c 10 http://localhost:8000/health 