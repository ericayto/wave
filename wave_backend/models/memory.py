"""
Memory and context management models.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, JSON, Text, LargeBinary, Integer
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional, Dict, Any

from .database import Base

class Event(Base):
    __tablename__ = "events"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # action, observation, decision
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

class MemorySummary(Base):
    __tablename__ = "memory_summaries"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    scope: Mapped[str] = mapped_column(String(20), nullable=False)  # minute, hour, day
    start_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # plan, critique, outcome
    ref_table: Mapped[str] = mapped_column(String(50), nullable=False)  # Referenced table
    ref_id: Mapped[int] = mapped_column(Integer, nullable=False)  # Referenced record ID
    vector_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # Serialized vector
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class PinnedFact(Base):
    __tablename__ = "pinned_facts"
    
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    ttl_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)  # Time to live
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class StateSnapshot(Base):
    __tablename__ = "state_snapshots"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    state_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256 hash of state