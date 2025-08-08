"""
WebSocket manager for real-time communication with frontend.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        # Active connections by connection ID
        self.connections: Dict[str, WebSocket] = {}
        # Subscriptions by topic
        self.subscriptions: Dict[str, Set[str]] = {}
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up WebSocket routes."""
        
        @self.router.websocket("/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
    
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        await websocket.accept()
        
        # Generate connection ID
        import uuid
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        
        logger.info(f"WebSocket connected: {connection_id}")
        
        try:
            # Send welcome message
            await websocket.send_text(json.dumps({
                "type": "connection",
                "data": {
                    "status": "connected",
                    "connection_id": connection_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }))
            
            # Handle messages
            async for message in websocket.iter_text():
                await self._handle_message(connection_id, message)
                
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await self.disconnect(connection_id)
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        # Remove from all subscriptions
        for topic, subscribers in self.subscriptions.items():
            subscribers.discard(connection_id)
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def _handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "subscribe":
                await self._handle_subscribe(connection_id, data)
            elif msg_type == "unsubscribe":
                await self._handle_unsubscribe(connection_id, data)
            elif msg_type == "ping":
                await self._handle_ping(connection_id)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_subscribe(self, connection_id: str, data: dict):
        """Handle subscription request."""
        topics = data.get("topics", [])
        
        for topic in topics:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(connection_id)
            logger.debug(f"Connection {connection_id} subscribed to {topic}")
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "subscribed",
            "data": {"topics": topics}
        })
    
    async def _handle_unsubscribe(self, connection_id: str, data: dict):
        """Handle unsubscription request."""
        topics = data.get("topics", [])
        
        for topic in topics:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(connection_id)
                logger.debug(f"Connection {connection_id} unsubscribed from {topic}")
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "unsubscribed",
            "data": {"topics": topics}
        })
    
    async def _handle_ping(self, connection_id: str):
        """Handle ping message."""
        await self.send_to_connection(connection_id, {
            "type": "pong",
            "data": {"timestamp": datetime.utcnow().isoformat()}
        })
    
    async def send_to_connection(self, connection_id: str, message: dict):
        """Send message to specific connection."""
        if connection_id not in self.connections:
            return
        
        try:
            websocket = self.connections[connection_id]
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def broadcast(self, topic: str, message: dict):
        """Broadcast message to all subscribers of a topic."""
        if topic not in self.subscriptions:
            return
        
        # Add metadata
        full_message = {
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat(),
            **message
        }
        
        # Send to all subscribers
        disconnected = []
        for connection_id in self.subscriptions[topic].copy():
            try:
                await self.send_to_connection(connection_id, full_message)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)
    
    async def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)
    
    async def get_subscription_count(self) -> Dict[str, int]:
        """Get subscription counts by topic."""
        return {topic: len(subscribers) for topic, subscribers in self.subscriptions.items()}