"""
User and authentication related models.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, Dict, Any

from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    exchanges: Mapped[list["Exchange"]] = relationship("Exchange", back_populates="user")

class Exchange(Base):
    __tablename__ = "exchanges"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(nullable=False, default=1)  # Single user for now
    name: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "kraken"
    cfg_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="exchanges")
    secrets: Mapped[list["Secret"]] = relationship("Secret", back_populates="exchange")
    positions: Mapped[list["Position"]] = relationship("Position", back_populates="exchange")
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="exchange")

class Secret(Base):
    __tablename__ = "secrets"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    exchange_id: Mapped[int] = mapped_column(nullable=False)
    key_alias: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "api_key", "api_secret"
    enc_blob: Mapped[str] = mapped_column(Text, nullable=False)  # Encrypted value
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    exchange: Mapped["Exchange"] = relationship("Exchange", back_populates="secrets")

# Import other models to ensure they're registered
from .trading import Position, Order
from .strategy import Strategy