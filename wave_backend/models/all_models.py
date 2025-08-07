"""
Import all models to ensure they are registered with SQLAlchemy.
"""

from .database import Base
from .user import User, Exchange, Secret
from .trading import Position, Order, Fill
from .strategy import Strategy, Backtest, Log, Plan, Metric
from .memory import Event, MemorySummary, Embedding, PinnedFact, StateSnapshot

__all__ = [
    "Base",
    "User", "Exchange", "Secret",
    "Position", "Order", "Fill", 
    "Strategy", "Backtest", "Log", "Plan", "Metric",
    "Event", "MemorySummary", "Embedding", "PinnedFact", "StateSnapshot"
]