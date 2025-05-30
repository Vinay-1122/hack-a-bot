#!/usr/bin/env python3
"""
Celery worker for the Multi-LLM Plotting Service.

This script starts a Celery worker that processes plotting jobs.
It can be run standalone or as part of a container orchestration system.
"""

import os
import sys
import logging
from celery import Celery

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from utils.logging_config import setup_logging
from core.job_manager import JobManager

# Setup logging
setup_logging(settings.LOG_LEVEL, json_logs=True)
logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    
    # Initialize Celery app
    celery_app = Celery(
        'plotting_service_worker',
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
        worker_disable_rate_limits=True,
        task_ignore_result=False,
        result_expires=3600,  # 1 hour
        worker_send_task_events=True,
        task_send_sent_event=True,
    )
    
    # Initialize job manager to register tasks
    job_manager = JobManager()
    
    logger.info("Celery worker application created successfully")
    return celery_app


# Create the Celery app
app = create_celery_app()


if __name__ == '__main__':
    """Run the Celery worker when script is executed directly."""
    
    logger.info("Starting Celery worker for Multi-LLM Plotting Service")
    
    # Start worker with configuration
    app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=4',
        '--hostname=plotting-worker@%h',
        '--queues=plotting',
        '--max-tasks-per-child=100',
        '--time-limit=600',  # 10 minutes hard limit
        '--soft-time-limit=300',  # 5 minutes soft limit
    ]) 