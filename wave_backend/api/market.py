"""
Market data endpoints.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random

router = APIRouter()

class MarketTicker(BaseModel):
    symbol: str
    price: float
    change_24h: float
    change_24h_pct: float
    volume_24h: float
    high_24h: float
    low_24h: float
    last_updated: datetime

class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketSummary(BaseModel):
    total_symbols: int
    active_pairs: List[str]
    last_updated: datetime

# Mock data for development
MOCK_TICKERS = {
    "BTC/USDT": MarketTicker(
        symbol="BTC/USDT",
        price=45000.0,
        change_24h=1200.0,
        change_24h_pct=2.74,
        volume_24h=1234567.89,
        high_24h=46500.0,
        low_24h=43200.0,
        last_updated=datetime.utcnow()
    ),
    "ETH/USDT": MarketTicker(
        symbol="ETH/USDT", 
        price=2800.0,
        change_24h=-45.0,
        change_24h_pct=-1.58,
        volume_24h=987654.32,
        high_24h=2950.0,
        low_24h=2750.0,
        last_updated=datetime.utcnow()
    )
}

@router.get("/summary", response_model=MarketSummary)
async def get_market_summary():
    """Get market overview."""
    return MarketSummary(
        total_symbols=len(MOCK_TICKERS),
        active_pairs=list(MOCK_TICKERS.keys()),
        last_updated=datetime.utcnow()
    )

@router.get("/ticker/{symbol}")
async def get_ticker(symbol: str):
    """Get ticker data for a symbol."""
    if symbol not in MOCK_TICKERS:
        # Generate mock data for any requested symbol
        base_price = random.uniform(100, 50000)
        change_pct = random.uniform(-5.0, 5.0)
        
        ticker = MarketTicker(
            symbol=symbol,
            price=base_price,
            change_24h=base_price * change_pct / 100,
            change_24h_pct=change_pct,
            volume_24h=random.uniform(10000, 1000000),
            high_24h=base_price * 1.05,
            low_24h=base_price * 0.95,
            last_updated=datetime.utcnow()
        )
        MOCK_TICKERS[symbol] = ticker
    
    return MOCK_TICKERS[symbol]

@router.get("/tickers")
async def get_tickers(symbols: Optional[str] = Query(None)):
    """Get ticker data for multiple symbols."""
    if symbols:
        symbol_list = symbols.split(",")
        result = {}
        for symbol in symbol_list:
            if symbol in MOCK_TICKERS:
                result[symbol] = MOCK_TICKERS[symbol]
        return {"tickers": result}
    
    return {"tickers": MOCK_TICKERS}

@router.get("/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    since: Optional[datetime] = None
):
    """Get OHLCV candlestick data."""
    # Generate mock OHLCV data
    end_time = datetime.utcnow()
    if since:
        end_time = since + timedelta(hours=limit)
    
    # Parse timeframe to minutes
    timeframe_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    interval_minutes = timeframe_minutes.get(timeframe, 60)
    
    ohlcv_data = []
    base_price = MOCK_TICKERS.get(symbol, MOCK_TICKERS["BTC/USDT"]).price
    
    for i in range(limit):
        timestamp = end_time - timedelta(minutes=interval_minutes * (limit - i - 1))
        
        # Generate realistic OHLCV data
        price_variation = random.uniform(0.98, 1.02)
        open_price = base_price * price_variation
        close_price = open_price * random.uniform(0.99, 1.01)
        high_price = max(open_price, close_price) * random.uniform(1.0, 1.005)
        low_price = min(open_price, close_price) * random.uniform(0.995, 1.0)
        volume = random.uniform(100, 10000)
        
        ohlcv_data.append(OHLCV(
            timestamp=timestamp,
            open=round(open_price, 2),
            high=round(high_price, 2), 
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=round(volume, 4)
        ))
        
        base_price = close_price  # Use close as next base
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": ohlcv_data
    }

@router.get("/orderbook/{symbol}")
async def get_orderbook(symbol: str, limit: int = 20):
    """Get order book data."""
    ticker = await get_ticker(symbol)
    base_price = ticker.price
    
    # Generate mock orderbook
    bids = []
    asks = []
    
    for i in range(limit):
        bid_price = base_price * (1 - (i + 1) * 0.001)
        ask_price = base_price * (1 + (i + 1) * 0.001)
        quantity = random.uniform(0.1, 10.0)
        
        bids.append([round(bid_price, 2), round(quantity, 4)])
        asks.append([round(ask_price, 2), round(quantity, 4)])
    
    return {
        "symbol": symbol,
        "bids": bids,  # [price, quantity]
        "asks": asks,  # [price, quantity] 
        "timestamp": datetime.utcnow()
    }