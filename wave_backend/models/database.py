"""
Database connection and initialization.
"""

import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from wave_backend.config.settings import get_settings

class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

# Global database instances
engine = None
SessionLocal = None

async def init_database():
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    
    settings = get_settings()
    
    # Convert SQLite URL to async
    database_url = settings.database.url
    if database_url.startswith("sqlite:///"):
        # Ensure data directory exists
        db_path = Path(database_url[10:])  # Remove "sqlite:///"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite+aiosqlite:///{db_path}"
    
    # Create async engine
    engine = create_async_engine(
        database_url,
        echo=settings.database.echo,
        future=True,
    )
    
    # Create session factory
    SessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Database initialized successfully!")

async def get_db() -> AsyncSession:
    """Get database session."""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def close_database():
    """Close database connections."""
    global engine
    if engine:
        await engine.dispose()
        print("✅ Database connections closed.")