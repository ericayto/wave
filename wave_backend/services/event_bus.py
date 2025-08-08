"""
Event Bus for pub/sub messaging system.
"""

import asyncio
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Event data class."""
    topic: str
    data: Any
    timestamp: datetime
    event_id: str

class EventBus:
    """Async event bus for pub/sub messaging."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        self.event_queue = asyncio.Queue()
        self.worker_task = None
    
    async def start(self):
        """Start the event bus."""
        if self.running:
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._event_worker())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus."""
        if not self.running:
            return
        
        self.running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        
        self.subscribers[topic].append(callback)
        logger.debug(f"Subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic."""
        if topic in self.subscribers:
            if callback in self.subscribers[topic]:
                self.subscribers[topic].remove(callback)
                logger.debug(f"Unsubscribed from topic: {topic}")
    
    async def publish(self, topic: str, data: Any, event_id: str = None):
        """Publish an event."""
        if not self.running:
            logger.warning("Event bus not running, dropping event")
            return
        
        import uuid
        if not event_id:
            event_id = str(uuid.uuid4())
        
        event = Event(
            topic=topic,
            data=data,
            timestamp=datetime.utcnow(),
            event_id=event_id
        )
        
        await self.event_queue.put(event)
        logger.debug(f"Published event: {topic}")
    
    async def _event_worker(self):
        """Worker coroutine that processes events."""
        while self.running:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._handle_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event):
        """Handle an event by notifying subscribers."""
        if event.topic not in self.subscribers:
            return
        
        tasks = []
        for callback in self.subscribers[event.topic]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback(event))
                else:
                    # Wrap sync callback in coroutine
                    task = asyncio.create_task(self._run_sync_callback(callback, event))
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating task for callback: {e}")
        
        # Wait for all callbacks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_sync_callback(self, callback: Callable, event: Event):
        """Run a synchronous callback in a thread pool."""
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, callback, event)

# Global event bus instance
_global_event_bus = None

def get_event_bus() -> EventBus:
    """Get or create the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus