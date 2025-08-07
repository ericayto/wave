"""
Strategy management endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

router = APIRouter()

class StrategyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive" 
    PAUSED = "paused"
    ERROR = "error"

class Strategy(BaseModel):
    id: str
    name: str
    version: str
    status: StrategyStatus
    description: str
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class CreateStrategyRequest(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

# Mock strategies
MOCK_STRATEGIES = [
    Strategy(
        id="sma-crossover-1",
        name="SMA Crossover",
        version="1.0.0",
        status=StrategyStatus.INACTIVE,
        description="Simple Moving Average crossover strategy",
        parameters={
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "fast_period": 20,
            "slow_period": 50,
            "position_size": 0.1
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    ),
    Strategy(
        id="rsi-mean-reversion-1", 
        name="RSI Mean Reversion",
        version="1.0.0",
        status=StrategyStatus.INACTIVE,
        description="RSI-based mean reversion strategy",
        parameters={
            "symbols": ["ETH/USDT"],
            "timeframe": "15m",
            "rsi_period": 14,
            "oversold_level": 30,
            "overbought_level": 70,
            "position_size": 0.1
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
]

@router.get("/", response_model=List[Strategy])
async def get_strategies():
    """Get all strategies."""
    return MOCK_STRATEGIES

@router.get("/{strategy_id}", response_model=Strategy)
async def get_strategy(strategy_id: str):
    """Get specific strategy."""
    for strategy in MOCK_STRATEGIES:
        if strategy.id == strategy_id:
            return strategy
    
    raise HTTPException(status_code=404, detail="Strategy not found")

@router.post("/", response_model=Strategy)
async def create_strategy(request: CreateStrategyRequest):
    """Create new strategy."""
    strategy = Strategy(
        id=f"custom-{len(MOCK_STRATEGIES) + 1}",
        name=request.name,
        version="1.0.0",
        status=StrategyStatus.INACTIVE,
        description=request.description,
        parameters=request.parameters,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    MOCK_STRATEGIES.append(strategy)
    return strategy

@router.put("/{strategy_id}/status")
async def update_strategy_status(strategy_id: str, status: StrategyStatus):
    """Update strategy status."""
    for i, strategy in enumerate(MOCK_STRATEGIES):
        if strategy.id == strategy_id:
            strategy.status = status
            strategy.updated_at = datetime.utcnow()
            MOCK_STRATEGIES[i] = strategy
            return {"message": f"Strategy {strategy_id} status updated to {status}"}
    
    raise HTTPException(status_code=404, detail="Strategy not found")

@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete strategy."""
    for i, strategy in enumerate(MOCK_STRATEGIES):
        if strategy.id == strategy_id:
            del MOCK_STRATEGIES[i]
            return {"message": f"Strategy {strategy_id} deleted"}
    
    raise HTTPException(status_code=404, detail="Strategy not found")
