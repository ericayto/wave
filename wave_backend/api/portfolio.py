"""
Portfolio and balance endpoints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime

router = APIRouter()

class Balance(BaseModel):
    symbol: str
    free: float
    used: float
    total: float

class Position(BaseModel):
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float

class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    invested_value: float
    daily_pnl: float
    daily_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float

class PortfolioResponse(BaseModel):
    summary: PortfolioSummary
    balances: List[Balance]
    positions: List[Position]
    last_updated: datetime

# Mock data for paper trading
MOCK_BALANCES = [
    Balance(symbol="USDT", free=10000.0, used=0.0, total=10000.0),
    Balance(symbol="BTC", free=0.0, used=0.0, total=0.0),
    Balance(symbol="ETH", free=0.0, used=0.0, total=0.0),
]

MOCK_POSITIONS = []

@router.get("/", response_model=PortfolioResponse)
async def get_portfolio():
    """Get current portfolio state."""
    # In paper trading mode, return mock data
    # TODO: Integrate with paper trading broker
    
    summary = PortfolioSummary(
        total_value=10000.0,
        cash_balance=10000.0,
        invested_value=0.0,
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        total_pnl=0.0,
        total_pnl_pct=0.0
    )
    
    return PortfolioResponse(
        summary=summary,
        balances=MOCK_BALANCES,
        positions=MOCK_POSITIONS,
        last_updated=datetime.utcnow()
    )

@router.get("/balances")
async def get_balances():
    """Get account balances."""
    return {"balances": MOCK_BALANCES}

@router.get("/positions")
async def get_positions():
    """Get current positions."""
    return {"positions": MOCK_POSITIONS}

@router.get("/pnl")
async def get_pnl(days: int = 7):
    """Get PnL history."""
    # Mock PnL data
    pnl_data = []
    base_date = datetime.utcnow()
    
    for i in range(days):
        date = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
        pnl_data.append({
            "date": date.isoformat(),
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "portfolio_value": 10000.0
        })
    
    return {"pnl_history": pnl_data}