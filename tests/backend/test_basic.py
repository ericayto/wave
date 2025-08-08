"""
Basic tests to verify test infrastructure works.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path  
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_python_version():
    """Test that we're running Python 3.11+"""
    assert sys.version_info >= (3, 11)


def test_project_structure():
    """Test that key directories exist."""
    wave_backend = PROJECT_ROOT / "wave_backend"
    wave_frontend = PROJECT_ROOT / "wave_frontend"
    tests_dir = PROJECT_ROOT / "tests"
    
    assert wave_backend.exists()
    assert wave_frontend.exists()
    assert tests_dir.exists()


def test_basic_calculations():
    """Test basic calculations work."""
    # Portfolio calculations
    balance = 10000.0
    position_value = 5000.0
    available = balance - position_value
    
    assert available == 5000.0
    
    # Percentage calculations
    profit = 150.0
    initial = 1000.0
    profit_pct = (profit / initial) * 100
    
    assert profit_pct == 15.0


def test_config_file_exists():
    """Test that config file exists."""
    config_file = PROJECT_ROOT / "config" / "wave.toml"
    assert config_file.exists()


def test_requirements_exist():
    """Test that requirements files exist."""
    backend_req = PROJECT_ROOT / "wave_backend" / "requirements.txt" 
    frontend_pkg = PROJECT_ROOT / "wave_frontend" / "package.json"
    
    assert backend_req.exists()
    assert frontend_pkg.exists()


def test_makefile_exists():
    """Test that Makefile exists."""
    makefile = PROJECT_ROOT / "Makefile"
    assert makefile.exists()


def test_imports_work():
    """Test basic imports work."""
    # Test that we can import basic Python modules
    import json
    import datetime
    import asyncio
    
    # Test JSON operations
    data = {"test": "value", "number": 42}
    json_str = json.dumps(data)
    parsed_data = json.loads(json_str)
    assert parsed_data["test"] == "value"
    
    # Test datetime operations
    now = datetime.datetime.now()
    assert isinstance(now, datetime.datetime)
    
    # Test asyncio
    assert asyncio is not None


@pytest.mark.asyncio
async def test_async_functionality():
    """Test that async/await works."""
    
    async def async_add(a, b):
        return a + b
    
    result = await async_add(2, 3)
    assert result == 5


def test_portfolio_calculations():
    """Test portfolio calculation logic."""
    
    class MockPortfolio:
        def __init__(self, initial_balance=10000):
            self.balance = initial_balance
            self.positions = {}
        
        def buy(self, symbol, qty, price):
            cost = qty * price
            if cost <= self.balance:
                self.balance -= cost
                if symbol not in self.positions:
                    self.positions[symbol] = {"qty": 0, "avg_price": 0}
                
                # Calculate new average price
                old_qty = self.positions[symbol]["qty"]
                old_avg = self.positions[symbol]["avg_price"]
                new_qty = old_qty + qty
                new_avg = ((old_qty * old_avg) + cost) / new_qty
                
                self.positions[symbol] = {"qty": new_qty, "avg_price": new_avg}
                return True
            return False
        
        def get_portfolio_value(self, current_prices):
            total = self.balance
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    total += position["qty"] * current_prices[symbol]
            return total
    
    # Test portfolio operations
    portfolio = MockPortfolio(10000)
    
    # Test buy operation
    success = portfolio.buy("BTC/USDT", 0.1, 50000)
    assert success is True
    assert portfolio.balance == 5000.0
    assert "BTC/USDT" in portfolio.positions
    assert portfolio.positions["BTC/USDT"]["qty"] == 0.1
    
    # Test portfolio value calculation
    current_prices = {"BTC/USDT": 52000}  # Price went up
    total_value = portfolio.get_portfolio_value(current_prices)
    expected_value = 5000 + (0.1 * 52000)  # cash + position value
    assert total_value == expected_value


def test_risk_calculations():
    """Test risk management calculations."""
    
    def calculate_position_size_limit(portfolio_value, max_position_pct):
        return portfolio_value * (max_position_pct / 100)
    
    def calculate_daily_loss_limit(portfolio_value, max_loss_pct):
        return portfolio_value * (max_loss_pct / 100)
    
    def is_position_size_valid(order_value, portfolio_value, max_pct):
        max_allowed = calculate_position_size_limit(portfolio_value, max_pct)
        return order_value <= max_allowed
    
    # Test position size limits
    portfolio_value = 10000
    max_position_pct = 25
    max_position_value = calculate_position_size_limit(portfolio_value, max_position_pct)
    assert max_position_value == 2500
    
    # Test order validation
    small_order = 1000
    large_order = 3000
    
    assert is_position_size_valid(small_order, portfolio_value, max_position_pct) is True
    assert is_position_size_valid(large_order, portfolio_value, max_position_pct) is False
    
    # Test daily loss limits
    daily_loss_limit = calculate_daily_loss_limit(portfolio_value, 2.0)
    assert daily_loss_limit == 200


def test_trading_calculations():
    """Test trading-related calculations."""
    
    def calculate_profit_loss(entry_price, current_price, quantity, side):
        if side == "buy":
            return (current_price - entry_price) * quantity
        else:  # sell/short
            return (entry_price - current_price) * quantity
    
    def calculate_percentage_return(entry_price, current_price):
        return ((current_price - entry_price) / entry_price) * 100
    
    # Test P&L calculations
    entry_price = 50000
    current_price = 52000
    quantity = 0.1
    
    pnl_long = calculate_profit_loss(entry_price, current_price, quantity, "buy")
    assert pnl_long == 200  # (52000 - 50000) * 0.1
    
    pnl_short = calculate_profit_loss(entry_price, current_price, quantity, "sell")
    assert pnl_short == -200  # (50000 - 52000) * 0.1
    
    # Test percentage returns
    pct_return = calculate_percentage_return(entry_price, current_price)
    assert pct_return == 4.0  # 4% gain


def test_strategy_validation():
    """Test strategy configuration validation."""
    
    def validate_strategy_config(config):
        required_fields = ["symbols", "timeframe"]
        errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate symbols
        if "symbols" in config:
            if not isinstance(config["symbols"], list) or len(config["symbols"]) == 0:
                errors.append("Symbols must be a non-empty list")
        
        # Validate timeframe
        if "timeframe" in config:
            valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            if config["timeframe"] not in valid_timeframes:
                errors.append(f"Invalid timeframe: {config['timeframe']}")
        
        return len(errors) == 0, errors
    
    # Test valid config
    valid_config = {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "1h"
    }
    is_valid, errors = validate_strategy_config(valid_config)
    assert is_valid is True
    assert len(errors) == 0
    
    # Test invalid config - missing symbols
    invalid_config = {
        "timeframe": "1h"
    }
    is_valid, errors = validate_strategy_config(invalid_config)
    assert is_valid is False
    assert len(errors) > 0
    
    # Test invalid config - invalid timeframe
    invalid_timeframe = {
        "symbols": ["BTC/USDT"],
        "timeframe": "invalid"
    }
    is_valid, errors = validate_strategy_config(invalid_timeframe)
    assert is_valid is False
    assert "timeframe" in str(errors)


if __name__ == "__main__":
    pytest.main([__file__])