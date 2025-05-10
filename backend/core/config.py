import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "analytics_db.sqlite")
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "schema_vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight but effective model

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Model pricing configuration
MODEL_PRICING_PER_MILLION_TOKENS = {
    "gemini-1.5-flash": {"input": 0.35, "output": 0.70},  # USD per 1M tokens
    "gemini-1.5-pro": {"input": 3.50, "output": 7.00},    # USD per 1M tokens
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},    # USD per 1M tokens
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8001
API_URL = f"http://127.0.0.1:{API_PORT}" 