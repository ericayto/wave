"""
Wave Backend API
Main FastAPI application entry point.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from wave_backend.config.settings import get_settings
from wave_backend.models.database import init_database
from wave_backend.models.all_models import *  # Import all models
from wave_backend.api import auth, portfolio, market, trading, strategies, risk, logs, memory, llm
from wave_backend.services.event_bus import EventBus
from wave_backend.services.websocket import WebSocketManager
from wave_backend.services.market_data import MarketDataService
from wave_backend.services.paper_broker import PaperBroker
from wave_backend.services.risk_engine import RiskEngine
from wave_backend.services.strategy_engine import StrategyEngine
from wave_backend.services.llm_planner import get_llm_planner
from wave_backend.strategies.sma_crossover import create_sma_crossover_strategy
from wave_backend.strategies.rsi_mean_reversion import create_rsi_mean_reversion_strategy

# Global instances
event_bus = EventBus()
websocket_manager = WebSocketManager()
market_data_service = MarketDataService(event_bus)
paper_broker = PaperBroker(event_bus)
risk_engine = RiskEngine(event_bus)
strategy_engine = StrategyEngine(event_bus, market_data_service, paper_broker, risk_engine)
llm_planner = None  # Initialized in lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🌊 Starting Wave Backend M1...")
    
    # Initialize database
    await init_database()
    
    # Start event bus
    await event_bus.start()
    
    # Start core services
    await market_data_service.start()
    await risk_engine.start()
    await strategy_engine.start()
    
    # Register default strategies
    sma_strategy = create_sma_crossover_strategy("sma_crossover_default", {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h"
    })
    await strategy_engine.register_strategy(sma_strategy)
    
    rsi_strategy = create_rsi_mean_reversion_strategy("rsi_mean_reversion_default", {
        "symbols": ["ETH/USDT"],
        "timeframe": "15m"
    })
    await strategy_engine.register_strategy(rsi_strategy)
    
    # Subscribe to market data for default symbols
    await market_data_service.subscribe_to_symbol("BTC/USDT")
    await market_data_service.subscribe_to_symbol("ETH/USDT")
    
    # Initialize LLM Planner (M2)
    global llm_planner
    settings = get_settings()
    if settings.llm.planning_enabled:
        llm_planner = get_llm_planner(
            user_id=1,  # Default user for now
            event_bus=event_bus,
            market_data=market_data_service,
            paper_broker=paper_broker,
            risk_engine=risk_engine,
            strategy_engine=strategy_engine
        )
        print("🧠 LLM Planner: Started with autonomous planning")
    else:
        print("🧠 LLM Planner: Available but autonomous planning disabled")
    
    print("✅ Wave Backend M2 started successfully!")
    print(f"   📊 Market Data: Connected")
    print(f"   🤖 Strategy Engine: {len(strategy_engine.strategies)} strategies registered") 
    print(f"   🛡️  Risk Engine: Active with limits enforcement")
    print(f"   💼 Paper Broker: Ready for realistic execution")
    print(f"   🧠 LLM Planner: {settings.llm.provider} provider with {settings.llm.model}")
    print(f"   💭 Context Manager: Ready for memory and RAG")
    print(f"   🎯 Strategy Generator: Ready for natural language strategies")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Wave Backend...")
    if llm_planner:
        await llm_planner.shutdown()
    await strategy_engine.stop()
    await risk_engine.stop()
    await market_data_service.stop()
    await event_bus.stop()
    print("✅ Wave Backend stopped.")

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Wave API",
        description="Local LLM-Driven Crypto Trading Bot",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
    app.include_router(market.router, prefix="/api/market", tags=["market"])
    app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
    app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
    app.include_router(risk.router, prefix="/api/risk", tags=["risk"])
    app.include_router(logs.router, prefix="/api/logs", tags=["logs"])
    app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
    app.include_router(llm.router, prefix="/api/llm", tags=["llm"])
    
    # WebSocket endpoint
    app.include_router(websocket_manager.router, prefix="/ws")
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}
    
    # Status endpoint
    @app.get("/status")
    async def get_status():
        llm_status = {}
        if llm_planner:
            llm_status = await llm_planner.get_status()
        
        return {
            "status": "running",
            "version": "M2",
            "mode": settings.core.mode,
            "services": {
                "market_data": {
                    "active_symbols": len(market_data_service.active_symbols),
                    "tickers_cached": len(market_data_service.tickers)
                },
                "strategy_engine": strategy_engine.get_engine_status(),
                "risk_engine": risk_engine.get_status(),
                "paper_broker": {
                    "portfolio_value": paper_broker.get_portfolio_value(),
                    "positions": len(paper_broker.get_positions()),
                    "pending_orders": len(paper_broker.orders)
                },
                "llm_planner": llm_status if llm_planner else {"status": "disabled"}
            },
            "llm": {
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "planning_enabled": settings.llm.planning_enabled,
                "planning_interval": settings.llm.planning_interval_seconds,
                "budgets": {
                    "hourly_tokens": settings.llm.hourly_token_budget,
                    "daily_tokens": settings.llm.daily_token_budget
                }
            }
        }
    
    return app

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=settings.ui.api_port,
        reload=True,
        log_level="info"
    )