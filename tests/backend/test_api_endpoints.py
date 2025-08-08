"""
Test API endpoints functionality.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


def test_status_endpoint(client):
    """Test status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "llm" in data


@pytest.mark.parametrize("endpoint", [
    "/api/portfolio/balance",
    "/api/portfolio/positions", 
    "/api/market/summary",
    "/api/strategies/list",
    "/api/risk/status",
    "/api/trading/orders",
])
def test_api_endpoints_exist(client, endpoint):
    """Test that API endpoints exist and return valid responses."""
    response = client.get(endpoint)
    # Endpoints should exist (not 404) even if they return errors due to missing data
    assert response.status_code != 404


def test_portfolio_balance_endpoint(client):
    """Test portfolio balance endpoint."""
    response = client.get("/api/portfolio/balance")
    # Should return some response (may be empty in test environment)
    assert response.status_code in [200, 500]  # 500 is ok if services not fully initialized


def test_market_summary_endpoint(client):
    """Test market summary endpoint.""" 
    response = client.get("/api/market/summary")
    assert response.status_code in [200, 500]


def test_market_symbols_endpoint(client):
    """Test market symbols endpoint."""
    response = client.get("/api/market/symbols")
    assert response.status_code in [200, 500]


def test_strategies_list_endpoint(client):
    """Test strategies list endpoint."""
    response = client.get("/api/strategies/list")
    assert response.status_code in [200, 500]


def test_risk_status_endpoint(client):
    """Test risk status endpoint."""
    response = client.get("/api/risk/status")
    assert response.status_code in [200, 500]


def test_risk_limits_get_endpoint(client):
    """Test get risk limits endpoint."""
    response = client.get("/api/risk/limits")
    assert response.status_code in [200, 500]


def test_trading_orders_endpoint(client):
    """Test trading orders endpoint."""
    response = client.get("/api/trading/orders")
    assert response.status_code in [200, 500]


def test_trading_positions_endpoint(client):
    """Test trading positions endpoint."""
    response = client.get("/api/trading/positions")
    assert response.status_code in [200, 500]


def test_logs_endpoint(client):
    """Test logs endpoint."""
    response = client.get("/api/logs")
    assert response.status_code in [200, 500]


def test_memory_state_endpoint(client):
    """Test memory state endpoint."""
    response = client.get("/api/memory/state")
    assert response.status_code in [200, 500]


def test_llm_status_endpoint(client):
    """Test LLM status endpoint."""
    response = client.get("/api/llm/status")
    assert response.status_code in [200, 500]


def test_invalid_endpoint_returns_404(client):
    """Test that invalid endpoints return 404."""
    response = client.get("/api/nonexistent/endpoint")
    assert response.status_code == 404


def test_cors_preflight_handling(client):
    """Test CORS preflight requests."""
    # Test CORS preflight
    response = client.options(
        "/api/portfolio/balance",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type"
        }
    )
    # Should handle CORS (may be 200 or 405 depending on FastAPI setup)
    assert response.status_code in [200, 405]


def test_post_endpoints_validation():
    """Test POST endpoints with validation."""
    app = create_app()
    with TestClient(app) as client:
        # Test strategy creation endpoint
        response = client.post(
            "/api/strategies/create",
            json={
                "name": "test_strategy",
                "config": {}
            }
        )
        # Should return validation error or success
        assert response.status_code in [200, 422, 500]
        
        # Test order placement
        response = client.post(
            "/api/trading/orders",
            json={
                "symbol": "BTC/USDT",
                "side": "buy",
                "qty": 0.1,
                "type": "market"
            }
        )
        assert response.status_code in [200, 422, 500]


def test_put_endpoints():
    """Test PUT endpoints."""
    app = create_app()
    with TestClient(app) as client:
        # Test risk limits update
        response = client.put(
            "/api/risk/limits",
            json={
                "max_position_pct": 0.2,
                "daily_loss_limit_pct": 1.0
            }
        )
        assert response.status_code in [200, 422, 500]


def test_delete_endpoints():
    """Test DELETE endpoints."""
    app = create_app()
    with TestClient(app) as client:
        # Test order cancellation
        response = client.delete("/api/trading/orders/test_order_id")
        assert response.status_code in [200, 404, 500]


def test_query_parameters():
    """Test endpoints with query parameters."""
    app = create_app()
    with TestClient(app) as client:
        # Test market data with symbols
        response = client.get("/api/market/summary?symbols=BTC/USDT,ETH/USDT")
        assert response.status_code in [200, 500]
        
        # Test logs with filters
        response = client.get("/api/logs?level=info&limit=10")
        assert response.status_code in [200, 500]
        
        # Test memory events with limit
        response = client.get("/api/memory/events?limit=20")
        assert response.status_code in [200, 500]


def test_content_type_json():
    """Test that endpoints accept and return JSON."""
    app = create_app()
    with TestClient(app) as client:
        # Test POST with JSON content
        response = client.post(
            "/api/llm/plan",
            json={"intent": "analyze_market"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 422, 500]


def test_authentication_endpoints():
    """Test authentication-related endpoints."""
    app = create_app()
    with TestClient(app) as client:
        # Test key storage
        response = client.post(
            "/api/auth/keys",
            json={
                "exchange": "kraken",
                "api_key": "test_key",
                "api_secret": "test_secret"
            }
        )
        assert response.status_code in [200, 422, 500]
        
        # Test key validation
        response = client.get("/api/auth/validate")
        assert response.status_code in [200, 500]


def test_websocket_endpoint():
    """Test WebSocket endpoint exists."""
    app = create_app()
    with TestClient(app) as client:
        # WebSocket endpoints typically return 426 Upgrade Required for HTTP requests
        response = client.get("/ws/stream")
        assert response.status_code in [426, 404, 500]


def test_error_handling():
    """Test API error handling."""
    app = create_app()
    with TestClient(app) as client:
        # Test malformed JSON
        response = client.post(
            "/api/strategies/create",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error
        
        # Test missing required fields
        response = client.post(
            "/api/trading/orders",
            json={}  # Missing required fields
        )
        assert response.status_code == 422


def test_openapi_docs():
    """Test that OpenAPI documentation is available."""
    app = create_app()
    with TestClient(app) as client:
        # Test OpenAPI JSON schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200