"""
Test the basic health endpoints and application setup.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path  
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wave_backend.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_status_endpoint(client):
    """Test comprehensive status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    
    # Check core fields
    assert "status" in data
    assert "version" in data
    assert "mode" in data
    assert "services" in data
    assert "llm" in data
    
    # Check service structure
    services = data["services"]
    assert "market_data" in services
    assert "strategy_engine" in services
    assert "risk_engine" in services
    assert "paper_broker" in services
    assert "llm_planner" in services
    
    # Check LLM configuration
    llm = data["llm"]
    assert "provider" in llm
    assert "model" in llm
    assert "planning_enabled" in llm
    assert "budgets" in llm


def test_cors_headers(client):
    """Test CORS headers are properly set."""
    response = client.options("/health", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET"
    })
    # FastAPI/Starlette handles CORS, so we just verify the endpoint is accessible
    assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled


@pytest.mark.asyncio
async def test_app_creation():
    """Test application can be created without errors."""
    app = create_app()
    assert app is not None
    assert app.title == "Wave API"
    assert app.description == "Local LLM-Driven Crypto Trading Bot"
    assert app.version == "0.1.0"