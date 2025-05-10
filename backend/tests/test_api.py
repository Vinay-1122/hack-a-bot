import pytest
from fastapi.testclient import TestClient
from ..api.app import app
import json

client = TestClient(app)

def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to HackaBot API" in response.json()["message"]

def test_query_analyzer_empty_db():
    """Test query analyzer with empty database."""
    payload = {
        "question": "What are the total sales?",
        "semantic_schema_json": "{}",
        "model_name": "gemini-1.5-flash",
        "use_rag_for_schema": False
    }
    response = client.post("/api/query-analyzer/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "error"
    assert "Database is empty" in data["thinking_steps"]

def test_semantic_schema_empty_db():
    """Test semantic schema generation with empty database."""
    payload = {
        "model_name": "gemini-1.5-flash"
    }
    response = client.post("/api/semantic-schema/", json=payload)
    assert response.status_code == 400
    assert "Database is empty" in response.json()["detail"]

def test_code_fix():
    """Test code fixing endpoint."""
    payload = {
        "code": "print('Hello, World!'",
        "error_message": "Missing parenthesis",
        "model_name": "gemini-1.5-flash"
    }
    response = client.post("/api/code-fix/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fixed_code" in data
    assert "explanation" in data 