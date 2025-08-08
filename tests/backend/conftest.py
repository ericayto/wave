"""
Pytest configuration and shared fixtures for backend tests.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
import sys

# Add project root to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure asyncio for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_config():
    """Create a temporary configuration file."""
    config_content = """
[core]
base_currency = "USD"
mode = "paper"

[risk]
max_position_pct = 0.25
daily_loss_limit_pct = 2.0
max_orders_per_hour = 6
circuit_breaker_spread_bps = 50

[llm]
provider = "mock"
model = "test-model"
planning_enabled = false
hourly_token_budget = 1000

[ui]
port = 5173
show_thinking_feed = true
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        return f.name


@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    return {
        "BTC/USDT": {
            "symbol": "BTC/USDT",
            "last": 50000.0,
            "bid": 49990.0,
            "ask": 50010.0,
            "volume": 1000.0,
            "high": 51000.0,
            "low": 49000.0,
            "change": 0.02
        },
        "ETH/USDT": {
            "symbol": "ETH/USDT", 
            "last": 3000.0,
            "bid": 2995.0,
            "ask": 3005.0,
            "volume": 5000.0,
            "high": 3100.0,
            "low": 2900.0,
            "change": 0.01
        }
    }


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio data for testing."""
    return {
        "total_value": 10000.0,
        "available_balance": 5000.0,
        "used_balance": 5000.0,
        "daily_pnl": 100.0,
        "total_pnl": 500.0,
        "positions": [
            {
                "symbol": "BTC/USDT",
                "qty": 0.1,
                "avg_price": 48000.0,
                "market_value": 5000.0,
                "unrealized_pnl": 200.0
            }
        ]
    }


# Test markers for different categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "async: async tests")
    config.addinivalue_line("markers", "slow: slow running tests")


# Skip tests that require external dependencies in CI
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    import pytest
    
    for item in items:
        # Mark async tests
        if "async" in item.name or any("async" in marker.name for marker in item.iter_markers()):
            item.add_marker(pytest.mark.asyncio)
            
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["stress", "performance", "load", "benchmark"]):
            item.add_marker(pytest.mark.slow)
            
        # Mark integration tests
        if any(keyword in item.name.lower() for keyword in ["integration", "e2e", "endpoint"]):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)