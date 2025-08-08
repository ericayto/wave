"""
Test database models and SQLAlchemy schema.
"""

import pytest
import tempfile
import asyncio
import sys
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wave_backend.models.database import Base, get_database_url, init_database
from wave_backend.models.user import User, Exchange, Secret
from wave_backend.models.trading import Position, Order, Fill
from wave_backend.models.strategy import Strategy, Backtest, Log, Plan, Metric
from wave_backend.models.memory import Event, MemorySummary, Embedding, PinnedFact, StateSnapshot


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    # Create temporary database file
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_file.close()
    
    # Create async engine for testing
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file.name}")
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session maker
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    yield engine, async_session
    
    # Cleanup
    await engine.dispose()
    Path(db_file.name).unlink()


@pytest.mark.asyncio
async def test_user_model(temp_db):
    """Test User model creation and queries."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create a user
        user = User(id=1)
        session.add(user)
        await session.commit()
        
        # Query the user
        result = await session.execute(select(User).where(User.id == 1))
        saved_user = result.scalar_one_or_none()
        
        assert saved_user is not None
        assert saved_user.id == 1
        assert saved_user.created_at is not None


@pytest.mark.asyncio
async def test_exchange_model(temp_db):
    """Test Exchange model."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create exchange
        exchange = Exchange(
            id=1,
            name="kraken",
            cfg_json='{"api_key": "test", "api_secret": "test"}'
        )
        session.add(exchange)
        await session.commit()
        
        # Query
        result = await session.execute(select(Exchange).where(Exchange.name == "kraken"))
        saved_exchange = result.scalar_one_or_none()
        
        assert saved_exchange is not None
        assert saved_exchange.name == "kraken"
        assert saved_exchange.cfg_json is not None


@pytest.mark.asyncio
async def test_position_model(temp_db):
    """Test Position model."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create position
        position = Position(
            id=1,
            exchange_id=1,
            symbol="BTC/USDT",
            qty=1.5,
            avg_price=50000.0,
            pnl_realized=100.0,
            pnl_unrealized=200.0
        )
        session.add(position)
        await session.commit()
        
        # Query
        result = await session.execute(select(Position).where(Position.symbol == "BTC/USDT"))
        saved_position = result.scalar_one_or_none()
        
        assert saved_position is not None
        assert saved_position.symbol == "BTC/USDT"
        assert saved_position.qty == 1.5
        assert saved_position.avg_price == 50000.0


@pytest.mark.asyncio
async def test_order_model(temp_db):
    """Test Order model."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create order
        order = Order(
            id="test_order_1",
            exchange_id=1,
            symbol="ETH/USDT",
            side="buy",
            qty=2.0,
            type="market",
            price=3000.0,
            status="filled"
        )
        session.add(order)
        await session.commit()
        
        # Query
        result = await session.execute(select(Order).where(Order.id == "test_order_1"))
        saved_order = result.scalar_one_or_none()
        
        assert saved_order is not None
        assert saved_order.symbol == "ETH/USDT"
        assert saved_order.side == "buy"
        assert saved_order.status == "filled"


@pytest.mark.asyncio
async def test_strategy_model(temp_db):
    """Test Strategy model."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create strategy
        strategy = Strategy(
            id="sma_crossover_test",
            name="SMA Crossover",
            version="1.0.0",
            json='{"signals": [], "entries": [], "exits": []}',
            status="active"
        )
        session.add(strategy)
        await session.commit()
        
        # Query
        result = await session.execute(select(Strategy).where(Strategy.id == "sma_crossover_test"))
        saved_strategy = result.scalar_one_or_none()
        
        assert saved_strategy is not None
        assert saved_strategy.name == "SMA Crossover"
        assert saved_strategy.status == "active"


@pytest.mark.asyncio
async def test_backtest_model(temp_db):
    """Test Backtest model."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create backtest
        from datetime import datetime, timezone
        
        backtest = Backtest(
            id=1,
            strategy_id="sma_crossover_test",
            params_json='{"period": 30}',
            window_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            window_end=datetime(2023, 12, 31, tzinfo=timezone.utc),
            metrics_json='{"sharpe_ratio": 1.5, "max_drawdown": 0.1}'
        )
        session.add(backtest)
        await session.commit()
        
        # Query
        result = await session.execute(select(Backtest).where(Backtest.id == 1))
        saved_backtest = result.scalar_one_or_none()
        
        assert saved_backtest is not None
        assert saved_backtest.strategy_id == "sma_crossover_test"
        assert saved_backtest.metrics_json is not None


@pytest.mark.asyncio
async def test_memory_models(temp_db):
    """Test memory-related models."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create event
        event = Event(
            id=1,
            type="order_placed",
            payload_json='{"symbol": "BTC/USDT", "side": "buy"}'
        )
        session.add(event)
        
        # Create memory summary
        from datetime import datetime, timezone
        summary = MemorySummary(
            id=1,
            scope="minute",
            start_ts=datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
            end_ts=datetime(2023, 1, 1, 12, 1, tzinfo=timezone.utc),
            tokens=150,
            text="Market was bullish, placed buy order for BTC"
        )
        session.add(summary)
        
        # Create pinned fact
        pinned_fact = PinnedFact(
            key="max_position_btc",
            value="2.0 BTC",
            ttl_ts=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        session.add(pinned_fact)
        
        await session.commit()
        
        # Query event
        result = await session.execute(select(Event).where(Event.type == "order_placed"))
        saved_event = result.scalar_one_or_none()
        assert saved_event is not None
        assert saved_event.type == "order_placed"
        
        # Query summary
        result = await session.execute(select(MemorySummary).where(MemorySummary.scope == "minute"))
        saved_summary = result.scalar_one_or_none()
        assert saved_summary is not None
        assert saved_summary.tokens == 150
        
        # Query pinned fact
        result = await session.execute(select(PinnedFact).where(PinnedFact.key == "max_position_btc"))
        saved_fact = result.scalar_one_or_none()
        assert saved_fact is not None
        assert saved_fact.value == "2.0 BTC"


@pytest.mark.asyncio
async def test_relationships(temp_db):
    """Test model relationships."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        # Create order and fill (relationship test)
        order = Order(
            id="order_with_fill",
            exchange_id=1,
            symbol="BTC/USDT",
            side="buy",
            qty=1.0,
            type="market",
            status="filled"
        )
        session.add(order)
        
        fill = Fill(
            id=1,
            order_id="order_with_fill",
            price=50000.0,
            qty=1.0,
            fee=25.0
        )
        session.add(fill)
        
        await session.commit()
        
        # Query order with fills
        result = await session.execute(
            select(Order).where(Order.id == "order_with_fill")
        )
        saved_order = result.scalar_one_or_none()
        
        assert saved_order is not None
        
        # Query the associated fill
        result = await session.execute(
            select(Fill).where(Fill.order_id == "order_with_fill")
        )
        saved_fill = result.scalar_one_or_none()
        
        assert saved_fill is not None
        assert saved_fill.price == 50000.0
        assert saved_fill.qty == 1.0


@pytest.mark.asyncio
async def test_log_model(temp_db):
    """Test Log model."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        log = Log(
            id=1,
            level="info",
            topic="strategy",
            message_json='{"message": "Strategy executed", "symbol": "BTC/USDT"}'
        )
        session.add(log)
        await session.commit()
        
        result = await session.execute(select(Log).where(Log.topic == "strategy"))
        saved_log = result.scalar_one_or_none()
        
        assert saved_log is not None
        assert saved_log.level == "info"
        assert saved_log.topic == "strategy"


@pytest.mark.asyncio
async def test_plan_model(temp_db):
    """Test Plan model for LLM planning."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        plan = Plan(
            id=1,
            intent="enter",
            rationale="RSI oversold, good entry opportunity",
            actions_json='[{"type": "place_order", "symbol": "BTC/USDT", "side": "buy"}]',
            risk_outcome="approved"
        )
        session.add(plan)
        await session.commit()
        
        result = await session.execute(select(Plan).where(Plan.intent == "enter"))
        saved_plan = result.scalar_one_or_none()
        
        assert saved_plan is not None
        assert saved_plan.intent == "enter"
        assert saved_plan.risk_outcome == "approved"


@pytest.mark.asyncio 
async def test_metric_model(temp_db):
    """Test Metric model."""
    engine, async_session = temp_db
    
    async with async_session() as session:
        metric = Metric(
            id=1,
            name="portfolio_value",
            value=10000.0,
            tags_json='{"currency": "USD", "type": "total"}'
        )
        session.add(metric)
        await session.commit()
        
        result = await session.execute(select(Metric).where(Metric.name == "portfolio_value"))
        saved_metric = result.scalar_one_or_none()
        
        assert saved_metric is not None
        assert saved_metric.value == 10000.0