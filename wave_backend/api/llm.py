"""
LLM API Endpoints

Provides API endpoints for LLM planning, strategy generation, and context management.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.llm_planner import get_llm_planner
from ..services.strategy_generator import get_strategy_generator
from ..services.context_manager import get_context_manager, EventType
from ..services.llm_orchestrator import get_orchestrator
from ..services.event_bus import get_event_bus
from ..services.market_data import MarketDataService
from ..services.paper_broker import PaperBroker
from ..services.risk_engine import RiskEngine
from ..services.strategy_engine import StrategyEngine


logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query or command")
    user_id: int = Field(default=1, description="User ID")


class StrategyGenerationRequest(BaseModel):
    description: str = Field(..., description="Natural language strategy description")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Additional constraints")
    user_id: int = Field(default=1, description="User ID")


class StrategyRefinementRequest(BaseModel):
    strategy_definition: Dict[str, Any] = Field(..., description="Current strategy definition")
    refinement_request: str = Field(..., description="Refinement request")
    user_id: int = Field(default=1, description="User ID")


class ContextQueryRequest(BaseModel):
    query: str = Field(..., description="Query for memory retrieval")
    k: Optional[int] = Field(default=6, description="Number of memories to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters for retrieval")
    user_id: int = Field(default=1, description="User ID")


class PinFactRequest(BaseModel):
    key: str = Field(..., description="Fact key")
    value: Any = Field(..., description="Fact value")
    ttl: Optional[int] = Field(default=None, description="Time to live in seconds")
    user_id: int = Field(default=1, description="User ID")


class PlannerControlRequest(BaseModel):
    action: str = Field(..., description="Control action: enable, disable, set_interval")
    value: Optional[Any] = Field(default=None, description="Value for the action")
    user_id: int = Field(default=1, description="User ID")


# Dependency injection helpers
async def get_services():
    """Get service instances for dependency injection."""
    # Import here to avoid circular imports
    from ..main import (
        event_bus, market_data_service, paper_broker, 
        risk_engine, strategy_engine, llm_planner
    )
    
    return {
        "event_bus": event_bus,
        "market_data": market_data_service,
        "paper_broker": paper_broker,
        "risk_engine": risk_engine,
        "strategy_engine": strategy_engine,
        "llm_planner": llm_planner
    }


@router.post("/query")
async def handle_llm_query(
    request: QueryRequest,
    services = Depends(get_services)
) -> Dict[str, Any]:
    """Handle natural language query or command."""
    try:
        llm_planner = services["llm_planner"]
        
        if not llm_planner:
            raise HTTPException(status_code=503, detail="LLM Planner not available")
        
        # Handle user query
        result = await llm_planner.handle_user_query(request.query)
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"LLM query handling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-strategy")
async def generate_strategy(request: StrategyGenerationRequest) -> Dict[str, Any]:
    """Generate trading strategy from natural language description."""
    try:
        generator = get_strategy_generator()
        
        result = await generator.generate_strategy(
            description=request.description,
            constraints=request.constraints,
            user_id=request.user_id
        )
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Strategy generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine-strategy")
async def refine_strategy(request: StrategyRefinementRequest) -> Dict[str, Any]:
    """Refine existing strategy based on user feedback."""
    try:
        generator = get_strategy_generator()
        
        result = await generator.refine_strategy(
            strategy_def=request.strategy_definition,
            refinement_request=request.refinement_request,
            user_id=request.user_id
        )
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Strategy refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/planner/status/{user_id}")
async def get_planner_status(
    user_id: int = 1,
    services = Depends(get_services)
) -> Dict[str, Any]:
    """Get LLM planner status."""
    try:
        llm_planner = services["llm_planner"]
        
        if not llm_planner:
            return {"status": "disabled", "message": "LLM Planner not initialized"}
        
        status = await llm_planner.get_status()
        return {
            "status": "success",
            "planner_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get planner status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/planner/control")
async def control_planner(
    request: PlannerControlRequest,
    services = Depends(get_services)
) -> Dict[str, Any]:
    """Control LLM planner (enable/disable/configure)."""
    try:
        llm_planner = services["llm_planner"]
        
        if not llm_planner:
            raise HTTPException(status_code=503, detail="LLM Planner not available")
        
        if request.action == "enable":
            llm_planner.enable_planning(True)
            message = "Planning enabled"
            
        elif request.action == "disable":
            llm_planner.enable_planning(False)
            message = "Planning disabled"
            
        elif request.action == "set_interval":
            if request.value is None or not isinstance(request.value, int):
                raise HTTPException(status_code=400, detail="set_interval requires integer value")
            llm_planner.set_planning_interval(request.value)
            message = f"Planning interval set to {request.value} seconds"
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
        
        return {
            "status": "success",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Planner control failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/state/{user_id}")
async def get_context_state(
    user_id: int = 1,
    keys: Optional[str] = None
) -> Dict[str, Any]:
    """Get current agent context state."""
    try:
        event_bus = get_event_bus()
        context_manager = get_context_manager(user_id, event_bus)
        
        key_list = keys.split(",") if keys else None
        state = await context_manager.get_context_state(key_list)
        
        return {
            "status": "success",
            "state": state,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get context state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/retrieve")
async def retrieve_memories(request: ContextQueryRequest) -> Dict[str, Any]:
    """Retrieve relevant memories using RAG."""
    try:
        event_bus = get_event_bus()
        context_manager = get_context_manager(request.user_id, event_bus)
        
        memories = await context_manager.retrieve_memories(
            query=request.query,
            k=request.k,
            filters=request.filters
        )
        
        return {
            "status": "success",
            "memories": memories,
            "count": len(memories),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/recent/{user_id}")
async def get_recent_events(user_id: int = 1, n: int = 10) -> Dict[str, Any]:
    """Get recent events."""
    try:
        event_bus = get_event_bus()
        context_manager = get_context_manager(user_id, event_bus)
        
        events = await context_manager.get_recent_events(n)
        
        return {
            "status": "success",
            "events": events,
            "count": len(events),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/pin-fact")
async def pin_fact(request: PinFactRequest) -> Dict[str, Any]:
    """Pin a critical fact to memory."""
    try:
        event_bus = get_event_bus()
        context_manager = get_context_manager(request.user_id, event_bus)
        
        await context_manager.pin_fact(
            key=request.key,
            value=request.value,
            ttl=request.ttl
        )
        
        return {
            "status": "success",
            "message": f"Fact '{request.key}' pinned successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to pin fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/context/pin-fact/{user_id}/{key}")
async def unpin_fact(user_id: int, key: str) -> Dict[str, Any]:
    """Remove a pinned fact."""
    try:
        event_bus = get_event_bus()
        context_manager = get_context_manager(user_id, event_bus)
        
        await context_manager.unpin_fact(key)
        
        return {
            "status": "success",
            "message": f"Fact '{key}' unpinned successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to unpin fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/pinned-facts/{user_id}")
async def get_pinned_facts(user_id: int = 1) -> Dict[str, Any]:
    """Get all pinned facts."""
    try:
        event_bus = get_event_bus()
        context_manager = get_context_manager(user_id, event_bus)
        
        facts = await context_manager.get_pinned_facts()
        
        return {
            "status": "success",
            "facts": facts,
            "count": len(facts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get pinned facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/stats/{user_id}")
async def get_memory_stats(user_id: int = 1) -> Dict[str, Any]:
    """Get memory system statistics."""
    try:
        event_bus = get_event_bus()
        context_manager = get_context_manager(user_id, event_bus)
        
        stats = context_manager.get_memory_stats()
        
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestrator/usage")
async def get_llm_usage() -> Dict[str, Any]:
    """Get LLM usage statistics."""
    try:
        orchestrator = get_orchestrator()
        
        stats = orchestrator.get_usage_stats()
        
        return {
            "status": "success",
            "usage_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get LLM usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrator/clear-cache")
async def clear_llm_cache() -> Dict[str, Any]:
    """Clear LLM response cache."""
    try:
        orchestrator = get_orchestrator()
        orchestrator.response_cache.clear()
        
        return {
            "status": "success",
            "message": "LLM cache cleared",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear LLM cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))