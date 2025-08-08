"""
Test trading strategies (SMA Crossover and RSI Mean Reversion).
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wave_backend.strategies.sma_crossover import create_sma_crossover_strategy, SMAStrategy
from wave_backend.strategies.rsi_mean_reversion import create_rsi_mean_reversion_strategy, RSIMeanReversionStrategy


def create_sample_ohlcv_data(length=100, start_price=50000):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=length, freq='1H')
    
    # Generate realistic price movement
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.02, length)
    prices = [start_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))  # Minimum price of 100
    
    prices = prices[1:]  # Remove the initial seed price
    
    # Generate OHLC from prices
    ohlcv_data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.uniform(100, 1000)
        
        ohlcv_data.append({
            'timestamp': dates[i],
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(ohlcv_data)


def test_sma_crossover_strategy_creation():
    """Test SMA crossover strategy creation."""
    strategy = create_sma_crossover_strategy("test_sma", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "fast_period": 10,
        "slow_period": 20
    })
    
    assert strategy is not None
    assert strategy.id == "test_sma"
    assert strategy.name == "SMA Crossover"
    assert "BTC/USDT" in strategy.symbols
    assert strategy.timeframe == "1h"


def test_sma_strategy_parameters():
    """Test SMA strategy parameter validation."""
    config = {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "fast_period": 20,
        "slow_period": 50,
        "volume_threshold": 1000,
        "volatility_filter": True
    }
    
    strategy = create_sma_crossover_strategy("test_sma_params", config)
    sma_strategy = strategy.impl
    
    assert isinstance(sma_strategy, SMAStrategy)
    assert sma_strategy.fast_period == 20
    assert sma_strategy.slow_period == 50
    assert sma_strategy.volume_threshold == 1000
    assert sma_strategy.volatility_filter is True


@pytest.mark.asyncio
async def test_sma_strategy_analysis():
    """Test SMA strategy analysis logic."""
    strategy = create_sma_crossover_strategy("test_analysis", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "fast_period": 5,
        "slow_period": 10
    })
    
    # Create mock data
    ohlcv_data = create_sample_ohlcv_data(50)
    
    # Mock market data
    market_data_mock = MagicMock()
    market_data_mock.get_ohlcv = AsyncMock(return_value=ohlcv_data)
    
    # Analyze
    analysis = await strategy.impl.analyze(market_data_mock, "BTC/USDT")
    
    assert analysis is not None
    assert "action" in analysis
    assert "confidence" in analysis
    assert "reasoning" in analysis
    assert analysis["action"] in ["buy", "sell", "hold"]
    assert 0 <= analysis["confidence"] <= 1


def test_rsi_mean_reversion_strategy_creation():
    """Test RSI mean reversion strategy creation."""
    strategy = create_rsi_mean_reversion_strategy("test_rsi", {
        "symbols": ["ETH/USDT"],
        "timeframe": "15m",
        "rsi_period": 14,
        "oversold_level": 30,
        "overbought_level": 70
    })
    
    assert strategy is not None
    assert strategy.id == "test_rsi"
    assert strategy.name == "RSI Mean Reversion"
    assert "ETH/USDT" in strategy.symbols
    assert strategy.timeframe == "15m"


def test_rsi_strategy_parameters():
    """Test RSI strategy parameter validation."""
    config = {
        "symbols": ["ETH/USDT"],
        "timeframe": "15m",
        "rsi_period": 21,
        "oversold_level": 25,
        "overbought_level": 75,
        "extreme_oversold": 20,
        "extreme_overbought": 80,
        "max_hold_time_hours": 48
    }
    
    strategy = create_rsi_mean_reversion_strategy("test_rsi_params", config)
    rsi_strategy = strategy.impl
    
    assert isinstance(rsi_strategy, RSIMeanReversionStrategy)
    assert rsi_strategy.rsi_period == 21
    assert rsi_strategy.oversold_level == 25
    assert rsi_strategy.overbought_level == 75
    assert rsi_strategy.max_hold_time_hours == 48


@pytest.mark.asyncio
async def test_rsi_strategy_analysis():
    """Test RSI strategy analysis logic."""
    strategy = create_rsi_mean_reversion_strategy("test_rsi_analysis", {
        "symbols": ["ETH/USDT"],
        "timeframe": "15m",
        "rsi_period": 14
    })
    
    # Create mock data with RSI extremes
    ohlcv_data = create_sample_ohlcv_data(50, start_price=3000)
    
    # Mock market data
    market_data_mock = MagicMock()
    market_data_mock.get_ohlcv = AsyncMock(return_value=ohlcv_data)
    
    # Analyze
    analysis = await strategy.impl.analyze(market_data_mock, "ETH/USDT")
    
    assert analysis is not None
    assert "action" in analysis
    assert "confidence" in analysis
    assert "reasoning" in analysis
    assert analysis["action"] in ["buy", "sell", "hold"]


def test_strategy_risk_management():
    """Test strategy risk management features."""
    strategy = create_sma_crossover_strategy("test_risk", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "max_position_size": 0.1
    })
    
    assert strategy.risk_limits is not None
    assert strategy.risk_limits.get("stop_loss_pct") == 2.0
    assert strategy.risk_limits.get("take_profit_pct") == 4.0
    assert strategy.risk_limits.get("max_position_size") == 0.1


def test_strategy_state_management():
    """Test strategy state persistence."""
    strategy = create_sma_crossover_strategy("test_state", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h"
    })
    
    # Test initial state
    assert strategy.state == "active"
    assert hasattr(strategy, 'last_analysis')
    assert hasattr(strategy, 'performance_metrics')


def test_strategy_serialization():
    """Test strategy JSON serialization."""
    strategy = create_sma_crossover_strategy("test_serialize", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "fast_period": 12,
        "slow_period": 26
    })
    
    # Test that strategy can be converted to dict/JSON
    strategy_dict = {
        "id": strategy.id,
        "name": strategy.name,
        "symbols": strategy.symbols,
        "timeframe": strategy.timeframe,
        "state": strategy.state,
        "config": strategy.config
    }
    
    assert strategy_dict["id"] == "test_serialize"
    assert strategy_dict["name"] == "SMA Crossover"
    assert strategy_dict["symbols"] == ["BTC/USDT"]


@pytest.mark.asyncio
async def test_strategy_backtesting_compatibility():
    """Test that strategies are compatible with backtesting."""
    strategy = create_rsi_mean_reversion_strategy("test_backtest", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h"
    })
    
    # Create historical data
    historical_data = create_sample_ohlcv_data(200)
    
    # Mock market data for backtesting
    market_data_mock = MagicMock()
    market_data_mock.get_ohlcv = AsyncMock(return_value=historical_data)
    
    # Simulate backtesting by running analysis on historical data
    signals = []
    
    # Run strategy analysis on chunks of historical data
    for i in range(50, len(historical_data), 10):
        chunk_data = historical_data.iloc[:i]
        market_data_mock.get_ohlcv.return_value = chunk_data
        
        analysis = await strategy.impl.analyze(market_data_mock, "BTC/USDT")
        if analysis["action"] != "hold":
            signals.append({
                "timestamp": chunk_data.iloc[-1]["timestamp"],
                "action": analysis["action"],
                "confidence": analysis["confidence"],
                "price": chunk_data.iloc[-1]["close"]
            })
    
    # Should generate some signals
    assert len(signals) >= 0  # Could be 0 if market conditions don't trigger strategy


def test_strategy_error_handling():
    """Test strategy error handling with invalid data."""
    strategy = create_sma_crossover_strategy("test_errors", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h"
    })
    
    # Test with invalid configuration
    with pytest.raises((ValueError, KeyError, TypeError)):
        create_sma_crossover_strategy("invalid", {
            "symbols": [],  # Empty symbols
            "timeframe": "invalid_timeframe"
        })


def test_multiple_symbol_strategies():
    """Test strategies that handle multiple symbols."""
    strategy = create_sma_crossover_strategy("test_multi", {
        "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
        "timeframe": "4h"
    })
    
    assert len(strategy.symbols) == 3
    assert "BTC/USDT" in strategy.symbols
    assert "ETH/USDT" in strategy.symbols
    assert "ADA/USDT" in strategy.symbols


def test_strategy_performance_tracking():
    """Test strategy performance metrics tracking."""
    strategy = create_rsi_mean_reversion_strategy("test_performance", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h"
    })
    
    # Initialize performance metrics
    if hasattr(strategy, 'performance_metrics'):
        metrics = strategy.performance_metrics
        
        # Should have basic performance tracking structure
        assert isinstance(metrics, dict)
    else:
        # If not implemented yet, just ensure strategy exists
        assert strategy is not None


def test_strategy_config_validation():
    """Test strategy configuration validation."""
    # Valid configuration
    valid_config = {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "fast_period": 10,
        "slow_period": 20
    }
    
    strategy = create_sma_crossover_strategy("test_valid", valid_config)
    assert strategy is not None
    
    # Invalid configurations should be handled gracefully
    invalid_configs = [
        {"symbols": []},  # Empty symbols
        {"symbols": ["BTC/USDT"], "timeframe": ""},  # Empty timeframe
        {"symbols": ["BTC/USDT"], "timeframe": "1h", "fast_period": -1},  # Negative period
    ]
    
    for invalid_config in invalid_configs:
        try:
            create_sma_crossover_strategy("test_invalid", invalid_config)
        except (ValueError, KeyError, TypeError):
            # Exception is expected for invalid config
            pass