"""
Memory and context management endpoints.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

router = APIRouter()

class StateProjection(BaseModel):
    keys: List[str]
    data: Dict[str, Any]
    timestamp: datetime

class Event(BaseModel):
    id: int
    timestamp: datetime
    type: str
    payload: Dict[str, Any]

class MemorySnippet(BaseModel):
    id: int
    relevance_score: float
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime

class PinnedFact(BaseModel):
    key: str
    value: str
    ttl: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class MemoryStats(BaseModel):
    total_events: int
    total_summaries: int
    total_pinned_facts: int
    context_usage_pct: float
    last_summary: datetime
    next_summary: datetime

# Mock data
MOCK_EVENTS = [
    Event(
        id=1,
        timestamp=datetime.utcnow(),
        type="decision",
        payload={"action": "hold", "reason": "insufficient signal strength"}
    ),
    Event(
        id=2,
        timestamp=datetime.utcnow() - timedelta(minutes=5),
        type="observation",
        payload={"market_condition": "sideways", "volatility": "low"}
    )
]

MOCK_PINNED_FACTS = [
    PinnedFact(
        key="max_position_btc",
        value="0.1 BTC",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
]

@router.get("/state")
async def get_context_state(keys: Optional[str] = Query(None)):
    """Get projection from agent state store."""
    if keys:
        key_list = keys.split(",")
    else:
        key_list = ["portfolio", "risk", "strategies"]
    
    # Mock state data
    state_data = {
        "portfolio": {
            "total_value": 10000.0,
            "cash_balance": 10000.0,
            "positions": []
        },
        "risk": {
            "exposure_pct": 0.0,
            "daily_loss_pct": 0.0,
            "status": "healthy"
        },
        "strategies": {
            "active_count": 0,
            "total_count": 2
        }
    }
    
    projection_data = {key: state_data.get(key, {}) for key in key_list}
    
    return StateProjection(
        keys=key_list,
        data=projection_data,
        timestamp=datetime.utcnow()
    )

@router.get("/events")
async def get_recent_events(n: int = Query(10, le=100)):
    """Get last N events."""
    # Sort by timestamp descending and limit
    sorted_events = sorted(MOCK_EVENTS, key=lambda x: x.timestamp, reverse=True)
    return {"events": sorted_events[:n]}

@router.get("/retrieve")
async def retrieve_memories(query: str, k: int = Query(5, le=20)):
    """Retrieve relevant memory snippets using RAG."""
    # Mock RAG retrieval
    mock_snippets = [
        MemorySnippet(
            id=1,
            relevance_score=0.85,
            content=f"Previous analysis showed {query} correlation with market volatility",
            metadata={"type": "analysis", "confidence": 0.8},
            timestamp=datetime.utcnow() - timedelta(hours=2)
        ),
        MemorySnippet(
            id=2,
            relevance_score=0.72,
            content=f"Strategy performed well during similar {query} conditions",
            metadata={"type": "strategy_outcome", "success": True},
            timestamp=datetime.utcnow() - timedelta(days=1)
        )
    ]
    
    return {"snippets": mock_snippets[:k]}

@router.post("/pin")
async def pin_fact(key: str, value: str, ttl: Optional[datetime] = None):
    """Pin a fact to critical facts registry."""
    fact = PinnedFact(
        key=key,
        value=value,
        ttl=ttl,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Update existing or add new
    for i, existing in enumerate(MOCK_PINNED_FACTS):
        if existing.key == key:
            MOCK_PINNED_FACTS[i] = fact
            return {"message": f"Fact '{key}' updated"}
    
    MOCK_PINNED_FACTS.append(fact)
    return {"message": f"Fact '{key}' pinned"}

@router.delete("/pin/{key}")
async def unpin_fact(key: str):
    """Remove a pinned fact."""
    global MOCK_PINNED_FACTS
    MOCK_PINNED_FACTS = [fact for fact in MOCK_PINNED_FACTS if fact.key != key]
    return {"message": f"Fact '{key}' unpinned"}

@router.get("/facts", response_model=List[PinnedFact])
async def get_pinned_facts():
    """Get all pinned facts."""
    # Filter out expired facts
    now = datetime.utcnow()
    active_facts = [
        fact for fact in MOCK_PINNED_FACTS
        if not fact.ttl or fact.ttl > now
    ]
    return active_facts

@router.post("/summarize")
async def force_summarize():
    """Force a memory summarization cycle."""
    return {
        "message": "Memory summarization initiated",
        "timestamp": datetime.utcnow(),
        "events_processed": len(MOCK_EVENTS)
    }

@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats():
    """Get memory system statistics."""
    return MemoryStats(
        total_events=len(MOCK_EVENTS),
        total_summaries=3,  # Mock
        total_pinned_facts=len(MOCK_PINNED_FACTS),
        context_usage_pct=15.5,  # Mock usage
        last_summary=datetime.utcnow() - timedelta(minutes=15),
        next_summary=datetime.utcnow() + timedelta(minutes=10)
    )
