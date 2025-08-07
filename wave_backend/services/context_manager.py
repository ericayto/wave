"""
Context Management System

Handles long-term memory, context windows, RAG retrieval, and state management
for LLM interactions. Implements hierarchical summarization and budget enforcement.
"""

import asyncio
import json
import logging
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import select, delete, update
from sqlalchemy.orm import selectinload

from ..config.settings import settings
from ..models.database import get_db
from ..models.memory import (
    Event, MemorySummary, Embedding, PinnedFact, StateSnapshot
)
from .event_bus import EventBus
from .llm_orchestrator import get_orchestrator, LLMMessage, MessageRole


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can be stored."""
    PLAN = "plan"
    ACTION = "action"
    OBSERVATION = "observation"
    DECISION = "decision"
    ERROR = "error"
    USER_INPUT = "user_input"


class SummaryScope(str, Enum):
    """Scopes for hierarchical summaries."""
    MINUTE = "minute"
    HOUR = "hour" 
    DAY = "day"
    WEEK = "week"


@dataclass
class ContextBudget:
    """Context window budget configuration."""
    max_context_tokens: int = 128000
    target_window_tokens: int = 24000
    rag_top_k: int = 6
    reserved_tokens: int = 2000  # Reserve for system prompt and tools


@dataclass
class AgentState:
    """Structured agent state (truth source)."""
    account_state: Dict[str, Any]
    strategy_state: Dict[str, Any] 
    risk_state: Dict[str, Any]
    run_state: Dict[str, Any]
    pinned_facts: Dict[str, Any]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "account_state": self.account_state,
            "strategy_state": self.strategy_state,
            "risk_state": self.risk_state,
            "run_state": self.run_state,
            "pinned_facts": self.pinned_facts,
            "last_updated": self.last_updated.isoformat()
        }


class ContextManager:
    """Central context and memory management system."""
    
    def __init__(self, event_bus: EventBus, user_id: int):
        self.event_bus = event_bus
        self.user_id = user_id
        self.budget = ContextBudget()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        self.embedding_cache: Dict[int, np.ndarray] = {}
        
        # Current agent state
        self.current_state: Optional[AgentState] = None
        
        # TF-IDF for keyword-based retrieval fallback
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_fitted = False
        
        # Initialize from database
        asyncio.create_task(self._initialize_from_db())
    
    async def _initialize_from_db(self):
        """Initialize context manager from database state."""
        try:
            async with get_db() as db:
                # Load latest state snapshot
                latest_snapshot = await db.execute(
                    select(StateSnapshot)
                    .where(StateSnapshot.user_id == self.user_id)
                    .order_by(StateSnapshot.ts.desc())
                    .limit(1)
                )
                snapshot = latest_snapshot.scalar_one_or_none()
                
                if snapshot:
                    state_data = json.loads(snapshot.state_json)
                    self.current_state = AgentState(
                        account_state=state_data.get('account_state', {}),
                        strategy_state=state_data.get('strategy_state', {}),
                        risk_state=state_data.get('risk_state', {}),
                        run_state=state_data.get('run_state', {}),
                        pinned_facts=state_data.get('pinned_facts', {}),
                        last_updated=datetime.fromisoformat(state_data['last_updated'])
                    )
                else:
                    # Initialize empty state
                    self.current_state = AgentState(
                        account_state={},
                        strategy_state={},
                        risk_state={},
                        run_state={"session_start": datetime.now().isoformat()},
                        pinned_facts={},
                        last_updated=datetime.now()
                    )
                
                # Load embeddings into FAISS index
                await self._load_embeddings()
                
        except Exception as e:
            logger.error(f"Failed to initialize context manager: {e}")
            # Fallback to empty state
            self.current_state = AgentState(
                account_state={},
                strategy_state={},
                risk_state={},
                run_state={"session_start": datetime.now().isoformat()},
                pinned_facts={},
                last_updated=datetime.now()
            )
    
    async def _load_embeddings(self):
        """Load embeddings from database into FAISS index."""
        try:
            async with get_db() as db:
                embeddings = await db.execute(
                    select(Embedding)
                    .where(Embedding.user_id == self.user_id)
                    .order_by(Embedding.id)
                )
                
                vectors = []
                for embedding in embeddings.scalars():
                    vector = np.frombuffer(embedding.vector_blob, dtype=np.float32)
                    vectors.append(vector)
                    self.embedding_cache[embedding.id] = vector
                
                if vectors:
                    vectors_array = np.array(vectors).astype('float32')
                    # Normalize vectors for cosine similarity
                    faiss.normalize_L2(vectors_array)
                    self.index.add(vectors_array)
                    
                    logger.info(f"Loaded {len(vectors)} embeddings into FAISS index")
                
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    
    async def record_event(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Record a new event."""
        try:
            async with get_db() as db:
                event = Event(
                    user_id=self.user_id,
                    ts=datetime.now(),
                    type=event_type.value,
                    payload_json=json.dumps(payload),
                    metadata_json=json.dumps(metadata or {})
                )
                db.add(event)
                await db.commit()
                await db.refresh(event)
                
                # Create embedding for the event
                await self._create_event_embedding(event)
                
                # Check if we need to trigger summarization
                await self._check_summarization_trigger()
                
                return event.id
                
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            raise
    
    async def _create_event_embedding(self, event: Event):
        """Create embedding for an event."""
        try:
            # Create text representation of event
            payload = json.loads(event.payload_json)
            text = f"{event.type}: {json.dumps(payload, default=str)}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            embedding = embedding.astype('float32')
            
            # Store in database
            async with get_db() as db:
                embedding_record = Embedding(
                    user_id=self.user_id,
                    kind="event",
                    ref_table="events",
                    ref_id=event.id,
                    vector_blob=embedding.tobytes(),
                    metadata_json=json.dumps({
                        "timestamp": event.ts.isoformat(),
                        "type": event.type
                    })
                )
                db.add(embedding_record)
                await db.commit()
                
                # Add to FAISS index
                faiss.normalize_L2(embedding.reshape(1, -1))
                self.index.add(embedding.reshape(1, -1))
                self.embedding_cache[embedding_record.id] = embedding
                
        except Exception as e:
            logger.error(f"Failed to create event embedding: {e}")
    
    async def get_context_state(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get projection of current agent state."""
        if not self.current_state:
            return {}
        
        state_dict = self.current_state.to_dict()
        
        if keys is None:
            return state_dict
        
        # Return only requested keys
        return {key: state_dict.get(key) for key in keys if key in state_dict}
    
    async def update_state(self, updates: Dict[str, Any]):
        """Update agent state with new information."""
        if not self.current_state:
            return
        
        # Update state fields
        for key, value in updates.items():
            if hasattr(self.current_state, key):
                setattr(self.current_state, key, value)
        
        self.current_state.last_updated = datetime.now()
        
        # Persist state snapshot
        await self._save_state_snapshot()
    
    async def _save_state_snapshot(self):
        """Save current state to database."""
        try:
            if not self.current_state:
                return
            
            state_json = json.dumps(self.current_state.to_dict(), default=str)
            state_hash = hashlib.md5(state_json.encode()).hexdigest()
            
            async with get_db() as db:
                snapshot = StateSnapshot(
                    user_id=self.user_id,
                    ts=datetime.now(),
                    state_json=state_json,
                    hash=state_hash
                )
                db.add(snapshot)
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")
    
    async def get_recent_events(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get last N events."""
        try:
            async with get_db() as db:
                events = await db.execute(
                    select(Event)
                    .where(Event.user_id == self.user_id)
                    .order_by(Event.ts.desc())
                    .limit(n)
                )
                
                return [
                    {
                        "id": event.id,
                        "timestamp": event.ts.isoformat(),
                        "type": event.type,
                        "payload": json.loads(event.payload_json),
                        "metadata": json.loads(event.metadata_json or "{}")
                    }
                    for event in events.scalars()
                ]
                
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    async def retrieve_memories(
        self, 
        query: str, 
        k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using RAG."""
        if k is None:
            k = self.budget.rag_top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            # Search FAISS index
            if self.index.ntotal == 0:
                return []
            
            scores, indices = self.index.search(query_embedding.reshape(1, -1), min(k * 2, self.index.ntotal))
            
            # Get corresponding events
            memories = []
            async with get_db() as db:
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1:  # No more results
                        break
                    
                    # Find embedding by index position
                    embedding = await db.execute(
                        select(Embedding)
                        .where(Embedding.user_id == self.user_id)
                        .offset(idx)
                        .limit(1)
                    )
                    embedding = embedding.scalar_one_or_none()
                    
                    if embedding and embedding.ref_table == "events":
                        # Get the event
                        event = await db.execute(
                            select(Event)
                            .where(Event.id == embedding.ref_id)
                        )
                        event = event.scalar_one_or_none()
                        
                        if event:
                            memories.append({
                                "id": event.id,
                                "timestamp": event.ts.isoformat(),
                                "type": event.type,
                                "payload": json.loads(event.payload_json),
                                "relevance_score": float(score),
                                "content": f"{event.type}: {json.dumps(json.loads(event.payload_json), default=str)}"
                            })
            
            # Sort by relevance and apply filters
            memories.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            if filters:
                # Apply simple filters
                if 'min_score' in filters:
                    memories = [m for m in memories if m['relevance_score'] >= filters['min_score']]
                if 'event_types' in filters:
                    memories = [m for m in memories if m['type'] in filters['event_types']]
                if 'since' in filters:
                    since_dt = datetime.fromisoformat(filters['since'])
                    memories = [m for m in memories if datetime.fromisoformat(m['timestamp']) >= since_dt]
            
            return memories[:k]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def pin_fact(self, key: str, value: Any, ttl: Optional[int] = None):
        """Pin a critical fact to the registry."""
        try:
            ttl_ts = None
            if ttl:
                ttl_ts = datetime.now() + timedelta(seconds=ttl)
            
            async with get_db() as db:
                # Upsert pinned fact
                fact = await db.execute(
                    select(PinnedFact)
                    .where(PinnedFact.user_id == self.user_id)
                    .where(PinnedFact.key == key)
                )
                fact = fact.scalar_one_or_none()
                
                if fact:
                    fact.value = json.dumps(value, default=str)
                    fact.ttl_ts = ttl_ts
                    fact.updated_at = datetime.now()
                else:
                    fact = PinnedFact(
                        user_id=self.user_id,
                        key=key,
                        value=json.dumps(value, default=str),
                        ttl_ts=ttl_ts
                    )
                    db.add(fact)
                
                await db.commit()
                
                # Update in-memory state
                if self.current_state:
                    self.current_state.pinned_facts[key] = value
                    await self._save_state_snapshot()
                
        except Exception as e:
            logger.error(f"Failed to pin fact: {e}")
            raise
    
    async def unpin_fact(self, key: str):
        """Remove a pinned fact."""
        try:
            async with get_db() as db:
                await db.execute(
                    delete(PinnedFact)
                    .where(PinnedFact.user_id == self.user_id)
                    .where(PinnedFact.key == key)
                )
                await db.commit()
                
                # Update in-memory state
                if self.current_state and key in self.current_state.pinned_facts:
                    del self.current_state.pinned_facts[key]
                    await self._save_state_snapshot()
                
        except Exception as e:
            logger.error(f"Failed to unpin fact: {e}")
            raise
    
    async def get_pinned_facts(self) -> Dict[str, Any]:
        """Get all active pinned facts."""
        try:
            async with get_db() as db:
                # Clean up expired facts first
                await db.execute(
                    delete(PinnedFact)
                    .where(PinnedFact.user_id == self.user_id)
                    .where(PinnedFact.ttl_ts.isnot(None))
                    .where(PinnedFact.ttl_ts < datetime.now())
                )
                
                # Get active facts
                facts = await db.execute(
                    select(PinnedFact)
                    .where(PinnedFact.user_id == self.user_id)
                    .where(
                        (PinnedFact.ttl_ts.is_(None)) |
                        (PinnedFact.ttl_ts > datetime.now())
                    )
                )
                
                return {
                    fact.key: json.loads(fact.value)
                    for fact in facts.scalars()
                }
                
        except Exception as e:
            logger.error(f"Failed to get pinned facts: {e}")
            return {}
    
    async def _check_summarization_trigger(self):
        """Check if we need to trigger summarization."""
        try:
            async with get_db() as db:
                # Count events since last summary
                last_summary = await db.execute(
                    select(MemorySummary)
                    .where(MemorySummary.user_id == self.user_id)
                    .where(MemorySummary.scope == SummaryScope.MINUTE.value)
                    .order_by(MemorySummary.end_ts.desc())
                    .limit(1)
                )
                last_summary = last_summary.scalar_one_or_none()
                
                since_ts = last_summary.end_ts if last_summary else datetime.now() - timedelta(hours=1)
                
                event_count = await db.execute(
                    select(Event)
                    .where(Event.user_id == self.user_id)
                    .where(Event.ts > since_ts)
                )
                event_count = len(event_count.scalars().all())
                
                # Trigger summarization if we have enough events
                summarize_every = getattr(settings.memory, 'summarize_every_events', 25)
                if event_count >= summarize_every:
                    await self._trigger_summarization(SummaryScope.MINUTE)
                
                # Also check time-based trigger
                time_diff = datetime.now() - since_ts
                summarize_every_minutes = getattr(settings.memory, 'summarize_every_minutes', 15)
                if time_diff.total_seconds() / 60 >= summarize_every_minutes:
                    await self._trigger_summarization(SummaryScope.MINUTE)
                
        except Exception as e:
            logger.error(f"Failed to check summarization trigger: {e}")
    
    async def _trigger_summarization(self, scope: SummaryScope):
        """Trigger summarization for a scope."""
        try:
            # Get events to summarize
            async with get_db() as db:
                # Find time window based on scope
                now = datetime.now()
                if scope == SummaryScope.MINUTE:
                    start_ts = now - timedelta(minutes=15)
                    end_ts = now
                elif scope == SummaryScope.HOUR:
                    start_ts = now - timedelta(hours=1)
                    end_ts = now
                elif scope == SummaryScope.DAY:
                    start_ts = now - timedelta(days=1)
                    end_ts = now
                else:
                    return
                
                events = await db.execute(
                    select(Event)
                    .where(Event.user_id == self.user_id)
                    .where(Event.ts >= start_ts)
                    .where(Event.ts <= end_ts)
                    .order_by(Event.ts)
                )
                events = events.scalars().all()
                
                if not events:
                    return
                
                # Create summary using LLM
                summary_text = await self._generate_summary(events, scope)
                
                if summary_text:
                    # Store summary
                    summary = MemorySummary(
                        user_id=self.user_id,
                        scope=scope.value,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        tokens=len(summary_text.split()),  # Rough token count
                        text=summary_text
                    )
                    db.add(summary)
                    await db.commit()
                    
                    # Create embedding for summary
                    await self._create_summary_embedding(summary)
                    
                    logger.info(f"Created {scope.value} summary covering {len(events)} events")
                
        except Exception as e:
            logger.error(f"Failed to trigger summarization: {e}")
    
    async def _generate_summary(self, events: List[Event], scope: SummaryScope) -> Optional[str]:
        """Generate summary using LLM."""
        try:
            # Build context from events
            event_texts = []
            for event in events:
                payload = json.loads(event.payload_json)
                event_texts.append(f"{event.ts.isoformat()} - {event.type}: {json.dumps(payload, default=str)}")
            
            context = "\n".join(event_texts)
            
            # Create summary prompt
            messages = [
                LLMMessage(
                    role=MessageRole.SYSTEM,
                    content=f"""You are a trading bot's memory system. Summarize the following {scope.value}-level events into a concise summary.

Focus on:
- Key decisions made
- Important market observations  
- Trading actions taken
- Risk events or violations
- Performance metrics

Keep the summary under 200 words and focus on actionable insights."""
                ),
                LLMMessage(
                    role=MessageRole.USER,
                    content=f"Summarize these events:\n\n{context}"
                )
            ]
            
            # Use local model for summarization to save costs
            orchestrator = get_orchestrator()
            from .llm_orchestrator import LLMProvider
            
            response = await orchestrator.generate(
                messages=messages,
                provider=LLMProvider.LOCAL,
                max_tokens=300,
                temperature=0.1
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
    
    async def _create_summary_embedding(self, summary: MemorySummary):
        """Create embedding for a summary."""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(summary.text)
            embedding = embedding.astype('float32')
            
            # Store in database
            async with get_db() as db:
                embedding_record = Embedding(
                    user_id=self.user_id,
                    kind="summary",
                    ref_table="memory_summaries",
                    ref_id=summary.id,
                    vector_blob=embedding.tobytes(),
                    metadata_json=json.dumps({
                        "scope": summary.scope,
                        "start_ts": summary.start_ts.isoformat(),
                        "end_ts": summary.end_ts.isoformat()
                    })
                )
                db.add(embedding_record)
                await db.commit()
                
                # Add to FAISS index
                faiss.normalize_L2(embedding.reshape(1, -1))
                self.index.add(embedding.reshape(1, -1))
                self.embedding_cache[embedding_record.id] = embedding
                
        except Exception as e:
            logger.error(f"Failed to create summary embedding: {e}")
    
    async def build_context_window(
        self, 
        query: str,
        include_recent: int = 5,
        include_rag: int = None
    ) -> Tuple[List[LLMMessage], Dict[str, Any]]:
        """
        Build optimized context window for LLM query.
        
        Returns:
            Tuple of (messages, metadata) where messages fit within budget
        """
        if include_rag is None:
            include_rag = self.budget.rag_top_k
        
        messages = []
        metadata = {
            "tokens_used": 0,
            "components": []
        }
        
        # Start with system prompt (always included)
        system_prompt = await self._build_system_prompt()
        messages.append(LLMMessage(role=MessageRole.SYSTEM, content=system_prompt))
        metadata["tokens_used"] += len(system_prompt.split()) * 1.3  # Rough token estimate
        
        # Add current state projection
        state = await self.get_context_state()
        if state:
            state_text = f"Current State:\n{json.dumps(state, indent=2, default=str)}"
            messages.append(LLMMessage(role=MessageRole.USER, content=state_text))
            metadata["tokens_used"] += len(state_text.split()) * 1.3
            metadata["components"].append("current_state")
        
        # Add pinned facts
        facts = await self.get_pinned_facts()
        if facts:
            facts_text = f"Pinned Facts:\n{json.dumps(facts, indent=2, default=str)}"
            messages.append(LLMMessage(role=MessageRole.USER, content=facts_text))
            metadata["tokens_used"] += len(facts_text.split()) * 1.3
            metadata["components"].append("pinned_facts")
        
        remaining_budget = self.budget.target_window_tokens - metadata["tokens_used"] - self.budget.reserved_tokens
        
        # Add RAG memories
        if remaining_budget > 1000 and include_rag > 0:
            memories = await self.retrieve_memories(query, include_rag)
            if memories:
                memory_texts = []
                for memory in memories:
                    memory_text = f"[{memory['timestamp']}] {memory['content']}"
                    if len(memory_text.split()) * 1.3 > remaining_budget:
                        break
                    memory_texts.append(memory_text)
                    remaining_budget -= len(memory_text.split()) * 1.3
                
                if memory_texts:
                    rag_content = f"Relevant Memory:\n" + "\n".join(memory_texts)
                    messages.append(LLMMessage(role=MessageRole.USER, content=rag_content))
                    metadata["tokens_used"] += len(rag_content.split()) * 1.3
                    metadata["components"].append("rag_memories")
        
        # Add recent events
        if remaining_budget > 500 and include_recent > 0:
            recent_events = await self.get_recent_events(include_recent)
            if recent_events:
                event_texts = []
                for event in reversed(recent_events):  # Chronological order
                    event_text = f"[{event['timestamp']}] {event['type']}: {json.dumps(event['payload'], default=str)}"
                    if len(event_text.split()) * 1.3 > remaining_budget:
                        break
                    event_texts.append(event_text)
                    remaining_budget -= len(event_text.split()) * 1.3
                
                if event_texts:
                    recent_content = f"Recent Events:\n" + "\n".join(event_texts)
                    messages.append(LLMMessage(role=MessageRole.USER, content=recent_content))
                    metadata["tokens_used"] += len(recent_content.split()) * 1.3
                    metadata["components"].append("recent_events")
        
        # Add the actual query
        messages.append(LLMMessage(role=MessageRole.USER, content=query))
        metadata["tokens_used"] += len(query.split()) * 1.3
        
        return messages, metadata
    
    async def _build_system_prompt(self) -> str:
        """Build system prompt with current configuration."""
        return """You are Wave, a sophisticated AI trading assistant. You help users analyze markets, manage risk, and make trading decisions.

Key capabilities:
- Analyze market data and generate trading insights
- Propose and backtest trading strategies  
- Monitor risk limits and portfolio exposure
- Execute paper trades with full risk validation
- Learn from past decisions to improve performance

Guidelines:
- Always validate trades through the risk engine before execution
- Focus on risk management and capital preservation
- Provide clear reasoning for all recommendations
- Use available tools to gather current market data
- Ask for human confirmation on significant decisions

You operate in paper trading mode only - no real money is at risk."""
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "faiss_index_size": self.index.ntotal,
            "embedding_cache_size": len(self.embedding_cache),
            "budget": {
                "max_context_tokens": self.budget.max_context_tokens,
                "target_window_tokens": self.budget.target_window_tokens,
                "rag_top_k": self.budget.rag_top_k
            },
            "current_state_age": (
                (datetime.now() - self.current_state.last_updated).total_seconds()
                if self.current_state else None
            )
        }


# Global context managers per user
_context_managers: Dict[int, ContextManager] = {}


def get_context_manager(user_id: int, event_bus: EventBus) -> ContextManager:
    """Get context manager for user."""
    global _context_managers
    if user_id not in _context_managers:
        _context_managers[user_id] = ContextManager(event_bus, user_id)
    return _context_managers[user_id]