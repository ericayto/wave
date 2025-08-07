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
            await websocket.send_text(json.dumps({\n                \"type\": \"connection\",\n                \"data\": {\n                    \"status\": \"connected\",\n                    \"connection_id\": connection_id,\n                    \"timestamp\": datetime.utcnow().isoformat()\n                }\n            }))\n            \n            # Handle messages\n            async for message in websocket.iter_text():\n                await self._handle_message(connection_id, message)\n                \n        except WebSocketDisconnect:\n            await self.disconnect(connection_id)\n        except Exception as e:\n            logger.error(f\"WebSocket error: {e}\")\n            await self.disconnect(connection_id)\n    \n    async def disconnect(self, connection_id: str):\n        \"\"\"Handle WebSocket disconnection.\"\"\"\n        if connection_id in self.connections:\n            del self.connections[connection_id]\n        \n        # Remove from all subscriptions\n        for topic, subscribers in self.subscriptions.items():\n            subscribers.discard(connection_id)\n        \n        logger.info(f\"WebSocket disconnected: {connection_id}\")\n    \n    async def _handle_message(self, connection_id: str, message: str):\n        \"\"\"Handle incoming WebSocket message.\"\"\"\n        try:\n            data = json.loads(message)\n            msg_type = data.get(\"type\")\n            \n            if msg_type == \"subscribe\":\n                await self._handle_subscribe(connection_id, data)\n            elif msg_type == \"unsubscribe\":\n                await self._handle_unsubscribe(connection_id, data)\n            elif msg_type == \"ping\":\n                await self._handle_ping(connection_id)\n            else:\n                logger.warning(f\"Unknown message type: {msg_type}\")\n                \n        except json.JSONDecodeError:\n            logger.error(f\"Invalid JSON message: {message}\")\n        except Exception as e:\n            logger.error(f\"Error handling message: {e}\")\n    \n    async def _handle_subscribe(self, connection_id: str, data: dict):\n        \"\"\"Handle subscription request.\"\"\"\n        topics = data.get(\"topics\", [])\n        \n        for topic in topics:\n            if topic not in self.subscriptions:\n                self.subscriptions[topic] = set()\n            \n            self.subscriptions[topic].add(connection_id)\n            logger.debug(f\"Connection {connection_id} subscribed to {topic}\")\n        \n        # Send confirmation\n        await self.send_to_connection(connection_id, {\n            \"type\": \"subscribed\",\n            \"data\": {\"topics\": topics}\n        })\n    \n    async def _handle_unsubscribe(self, connection_id: str, data: dict):\n        \"\"\"Handle unsubscription request.\"\"\"\n        topics = data.get(\"topics\", [])\n        \n        for topic in topics:\n            if topic in self.subscriptions:\n                self.subscriptions[topic].discard(connection_id)\n                logger.debug(f\"Connection {connection_id} unsubscribed from {topic}\")\n        \n        # Send confirmation\n        await self.send_to_connection(connection_id, {\n            \"type\": \"unsubscribed\",\n            \"data\": {\"topics\": topics}\n        })\n    \n    async def _handle_ping(self, connection_id: str):\n        \"\"\"Handle ping message.\"\"\"\n        await self.send_to_connection(connection_id, {\n            \"type\": \"pong\",\n            \"data\": {\"timestamp\": datetime.utcnow().isoformat()}\n        })\n    \n    async def send_to_connection(self, connection_id: str, message: dict):\n        \"\"\"Send message to specific connection.\"\"\"\n        if connection_id not in self.connections:\n            return\n        \n        try:\n            websocket = self.connections[connection_id]\n            await websocket.send_text(json.dumps(message))\n        except Exception as e:\n            logger.error(f\"Error sending to connection {connection_id}: {e}\")\n            await self.disconnect(connection_id)\n    \n    async def broadcast(self, topic: str, message: dict):\n        \"\"\"Broadcast message to all subscribers of a topic.\"\"\"\n        if topic not in self.subscriptions:\n            return\n        \n        # Add metadata\n        full_message = {\n            \"topic\": topic,\n            \"timestamp\": datetime.utcnow().isoformat(),\n            **message\n        }\n        \n        # Send to all subscribers\n        disconnected = []\n        for connection_id in self.subscriptions[topic].copy():\n            try:\n                await self.send_to_connection(connection_id, full_message)\n            except Exception as e:\n                logger.error(f\"Error broadcasting to {connection_id}: {e}\")\n                disconnected.append(connection_id)\n        \n        # Clean up disconnected connections\n        for connection_id in disconnected:\n            await self.disconnect(connection_id)\n    \n    async def get_connection_count(self) -> int:\n        \"\"\"Get number of active connections.\"\"\"\n        return len(self.connections)\n    \n    async def get_subscription_count(self) -> Dict[str, int]:\n        \"\"\"Get subscription counts by topic.\"\"\"\n        return {topic: len(subscribers) for topic, subscribers in self.subscriptions.items()}