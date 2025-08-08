"""
Test the EventBus service for pub/sub messaging.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wave_backend.services.event_bus import EventBus


@pytest.mark.asyncio
async def test_event_bus_creation():
    """Test EventBus can be created."""
    event_bus = EventBus()
    assert event_bus is not None
    assert event_bus.topics == {}
    assert not event_bus.is_running


@pytest.mark.asyncio
async def test_event_bus_lifecycle():
    """Test EventBus start and stop."""
    event_bus = EventBus()
    
    # Test start
    await event_bus.start()
    assert event_bus.is_running
    
    # Test stop
    await event_bus.stop()
    assert not event_bus.is_running


@pytest.mark.asyncio
async def test_event_subscription_and_publishing():
    """Test subscribing to and publishing events."""
    event_bus = EventBus()
    await event_bus.start()
    
    # Track received events
    received_events = []
    
    async def event_handler(event):
        received_events.append(event)
    
    try:
        # Subscribe to a topic
        await event_bus.subscribe("test_topic", event_handler)
        assert "test_topic" in event_bus.topics
        assert len(event_bus.topics["test_topic"]) == 1
        
        # Publish an event
        test_event = {"type": "test", "data": "hello world"}
        await event_bus.publish("test_topic", test_event)
        
        # Give some time for async processing
        await asyncio.sleep(0.1)
        
        # Check event was received
        assert len(received_events) == 1
        assert received_events[0] == test_event
        
    finally:
        await event_bus.stop()


@pytest.mark.asyncio
async def test_multiple_subscribers():
    """Test multiple subscribers to the same topic."""
    event_bus = EventBus()
    await event_bus.start()
    
    received_events_1 = []
    received_events_2 = []
    
    async def handler_1(event):
        received_events_1.append(event)
    
    async def handler_2(event):
        received_events_2.append(event)
    
    try:
        # Subscribe both handlers
        await event_bus.subscribe("multi_topic", handler_1)
        await event_bus.subscribe("multi_topic", handler_2)
        
        # Publish event
        test_event = {"type": "multi_test", "data": 123}
        await event_bus.publish("multi_topic", test_event)
        
        # Give time for processing
        await asyncio.sleep(0.1)
        
        # Both handlers should receive the event
        assert len(received_events_1) == 1
        assert len(received_events_2) == 1
        assert received_events_1[0] == test_event
        assert received_events_2[0] == test_event
        
    finally:
        await event_bus.stop()


@pytest.mark.asyncio
async def test_unsubscribe():
    """Test unsubscribing from events."""
    event_bus = EventBus()
    await event_bus.start()
    
    received_events = []
    
    async def event_handler(event):
        received_events.append(event)
    
    try:
        # Subscribe and publish
        await event_bus.subscribe("unsub_topic", event_handler)
        await event_bus.publish("unsub_topic", {"test": 1})
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        
        # Unsubscribe
        await event_bus.unsubscribe("unsub_topic", event_handler)
        
        # Publish again - should not be received
        await event_bus.publish("unsub_topic", {"test": 2})
        await asyncio.sleep(0.1)
        
        # Should still be 1
        assert len(received_events) == 1
        
    finally:
        await event_bus.stop()


@pytest.mark.asyncio
async def test_publishing_to_nonexistent_topic():
    """Test publishing to a topic with no subscribers."""
    event_bus = EventBus()
    await event_bus.start()
    
    try:
        # This should not raise an exception
        await event_bus.publish("nonexistent_topic", {"test": "data"})
        
    finally:
        await event_bus.stop()


@pytest.mark.asyncio
async def test_handler_exception_handling():
    """Test that handler exceptions don't crash the event bus."""
    event_bus = EventBus()
    await event_bus.start()
    
    received_events = []
    
    async def failing_handler(event):
        raise Exception("Handler failed")
    
    async def working_handler(event):
        received_events.append(event)
    
    try:
        # Subscribe both handlers
        await event_bus.subscribe("error_topic", failing_handler)
        await event_bus.subscribe("error_topic", working_handler)
        
        # Publish event
        test_event = {"test": "error_handling"}
        await event_bus.publish("error_topic", test_event)
        await asyncio.sleep(0.1)
        
        # Working handler should still receive the event
        assert len(received_events) == 1
        assert received_events[0] == test_event
        
    finally:
        await event_bus.stop()