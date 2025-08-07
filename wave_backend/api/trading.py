"""
Trading and order management endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter()

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

class PlaceOrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    quantity: float
    type: OrderType
    price: Optional[float] = None
    time_in_force: str = "GTC"

class Order(BaseModel):
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    type: OrderType
    price: Optional[float]
    avg_fill_price: Optional[float]
    status: OrderStatus
    created_at: datetime
    updated_at: datetime

class Fill(BaseModel):
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: datetime

class ExecuteResponse(BaseModel):
    success: bool
    order: Optional[Order] = None
    message: str

# Mock data storage
MOCK_ORDERS: Dict[str, Order] = {}
MOCK_FILLS: List[Fill] = []

@router.post("/execute", response_model=ExecuteResponse)
async def execute_plan():
    """Execute a trading plan (paper trading only)."""
    # TODO: Implement plan execution logic
    return ExecuteResponse(
        success=True,
        message="Plan execution not yet implemented - paper trading mode"
    )

@router.post("/orders", response_model=ExecuteResponse)
async def place_order(request: PlaceOrderRequest):
    """Place a new order (paper trading only)."""
    # Validate order
    if request.type == OrderType.LIMIT and not request.price:
        raise HTTPException(status_code=400, detail="Limit orders require a price")
    
    # Create order
    order_id = str(uuid.uuid4())
    order = Order(
        id=order_id,
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        filled_quantity=0.0,
        type=request.type,
        price=request.price,
        avg_fill_price=None,
        status=OrderStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Store order
    MOCK_ORDERS[order_id] = order
    
    # Simulate immediate fill for market orders (paper trading)
    if request.type == OrderType.MARKET:
        # Get mock price from market data
        mock_price = 45000.0 if "BTC" in request.symbol else 2800.0
        
        # Create fill
        fill = Fill(
            id=str(uuid.uuid4()),
            order_id=order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=mock_price,
            fee=request.quantity * mock_price * 0.001,  # 0.1% fee
            timestamp=datetime.utcnow()
        )
        
        MOCK_FILLS.append(fill)
        
        # Update order
        order.filled_quantity = request.quantity
        order.avg_fill_price = mock_price
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.utcnow()
        MOCK_ORDERS[order_id] = order
    
    return ExecuteResponse(
        success=True,
        order=order,
        message=f"Order placed successfully: {order_id}"
    )

@router.get("/orders")
async def get_orders(status: Optional[OrderStatus] = None):
    """Get order history."""
    orders = list(MOCK_ORDERS.values())
    
    if status:
        orders = [order for order in orders if order.status == status]
    
    # Sort by created_at descending
    orders.sort(key=lambda x: x.created_at, reverse=True)
    
    return {"orders": orders}

@router.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get specific order."""
    if order_id not in MOCK_ORDERS:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return MOCK_ORDERS[order_id]

@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order."""
    if order_id not in MOCK_ORDERS:
        raise HTTPException(status_code=404, detail="Order not found")
    
    order = MOCK_ORDERS[order_id]
    
    if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
        raise HTTPException(status_code=400, detail="Order cannot be canceled")
    
    order.status = OrderStatus.CANCELED
    order.updated_at = datetime.utcnow()
    MOCK_ORDERS[order_id] = order
    
    return {"message": f"Order {order_id} canceled successfully"}

@router.get("/fills")
async def get_fills(symbol: Optional[str] = None):
    """Get trade fills."""
    fills = MOCK_FILLS
    
    if symbol:
        fills = [fill for fill in fills if fill.symbol == symbol]
    
    # Sort by timestamp descending
    fills.sort(key=lambda x: x.timestamp, reverse=True)
    
    return {"fills": fills}

@router.get("/positions")
async def get_positions():
    """Get current positions (calculated from fills)."""
    positions = {}
    
    for fill in MOCK_FILLS:
        symbol = fill.symbol
        if symbol not in positions:
            positions[symbol] = {
                "symbol": symbol,
                "quantity": 0.0,
                "avg_price": 0.0,
                "total_cost": 0.0
            }
        
        position = positions[symbol]
        
        if fill.side == OrderSide.BUY:
            old_qty = position["quantity"]
            old_cost = position["total_cost"]
            new_qty = old_qty + fill.quantity
            new_cost = old_cost + (fill.quantity * fill.price)
            
            position["quantity"] = new_qty
            position["total_cost"] = new_cost
            position["avg_price"] = new_cost / new_qty if new_qty > 0 else 0.0
        else:  # SELL
            position["quantity"] -= fill.quantity
            if position["quantity"] < 0:
                position["quantity"] = 0.0
    
    # Filter out zero positions
    active_positions = {k: v for k, v in positions.items() if v["quantity"] > 0.001}
    
    return {"positions": list(active_positions.values())}