"""
Trading related models: positions, orders, fills.
"""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import String, DateTime, Numeric, Integer, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional
from enum import Enum

from .database import Base

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

class Position(Base):
    __tablename__ = "positions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    exchange_id: Mapped[int] = mapped_column(ForeignKey("exchanges.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., "BTC/USDT"
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    avg_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    pnl_realized: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    pnl_unrealized: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exchange: Mapped["Exchange"] = relationship("Exchange", back_populates="positions")

class Order(Base):
    __tablename__ = "orders"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    exchange_id: Mapped[int] = mapped_column(ForeignKey("exchanges.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[OrderSide] = mapped_column(String(10), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    type: Mapped[OrderType] = mapped_column(String(20), nullable=False)
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))  # NULL for market orders
    status: Mapped[OrderStatus] = mapped_column(String(20), nullable=False, default=OrderStatus.PENDING)
    filled_qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    avg_fill_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    
    # Metadata
    client_order_id: Mapped[Optional[str]] = mapped_column(String(100))
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exchange: Mapped["Exchange"] = relationship("Exchange", back_populates="orders")
    fills: Mapped[list["Fill"]] = relationship("Fill", back_populates="order")

class Fill(Base):
    __tablename__ = "fills"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Exchange metadata
    trade_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="fills")

# Forward references for imports
from .user import Exchange