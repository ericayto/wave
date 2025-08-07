"""
Strategy and backtesting related models.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, JSON, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, Dict, Any
from enum import Enum

from .database import Base

class StrategyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"

class Strategy(Base):
    __tablename__ = "strategies"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)  # Strategy definition
    status: Mapped[StrategyStatus] = mapped_column(String(20), nullable=False, default=StrategyStatus.INACTIVE)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    backtests: Mapped[list["Backtest"]] = relationship("Backtest", back_populates="strategy")

class Backtest(Base):
    __tablename__ = "backtests"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(nullable=False)
    params_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metrics_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Status and metadata
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="backtests")

class Log(Base):
    __tablename__ = "logs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    level: Mapped[str] = mapped_column(String(10), nullable=False)  # DEBUG, INFO, WARNING, ERROR
    topic: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "trading", "strategy", "risk"
    message_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

class Plan(Base):
    __tablename__ = "plans"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    intent: Mapped[str] = mapped_column(String(50), nullable=False)  # rebalance, enter, exit, hold, tune_strategy
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    actions_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    risk_outcome: Mapped[str] = mapped_column(String(20), nullable=False)  # approved, rejected, modified

class Metric(Base):
    __tablename__ = "metrics"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float] = mapped_column(nullable=False)
    tags_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)