"""
Enhanced Paper Trading Broker
Realistic order execution with slippage, fees, and market impact simulation.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import logging

from ..models.trading import Order, OrderSide, OrderType, OrderStatus, Fill
from ..services.event_bus import EventBus

logger = logging.getLogger(__name__)

class ExecutionModel(str, Enum):
    IMMEDIATE = "immediate"
    REALISTIC = "realistic"
    CONSERVATIVE = "conservative"

@dataclass
class MarketData:
    """Current market data for a symbol."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime

@dataclass
class OrderBookLevel:
    """Order book price level."""
    price: float
    quantity: float

@dataclass
class OrderBook:
    """Simulated order book."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime

class PaperBroker:
    """Enhanced paper trading broker with realistic execution."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.positions: Dict[str, Dict] = {}
        
        # Execution settings
        self.execution_model = ExecutionModel.REALISTIC
        self.base_fee_rate = 0.001  # 0.1% base fee
        self.slippage_factor = 0.0005  # 0.05% base slippage
        self.latency_range = (50, 200)  # Execution latency in ms
        
        # Market data cache
        self.market_data: Dict[str, MarketData] = {}
        self.order_books: Dict[str, OrderBook] = {}
        
        # Initialize with some mock data
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with mock market data."""
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"]
        
        for symbol in symbols:
            # Generate realistic price based on symbol
            if "BTC" in symbol:
                base_price = 45000.0
            elif "ETH" in symbol:
                base_price = 2800.0
            elif "SOL" in symbol:
                base_price = 95.0
            else:
                base_price = 0.5
            
            # Add some random variation
            price_variation = random.uniform(0.95, 1.05)
            current_price = base_price * price_variation
            
            # Create bid/ask spread (typical 0.01-0.05%)
            spread_pct = random.uniform(0.0001, 0.0005)
            spread = current_price * spread_pct
            
            self.market_data[symbol] = MarketData(
                symbol=symbol,
                bid=current_price - spread/2,
                ask=current_price + spread/2,
                last=current_price,
                volume_24h=random.uniform(100000, 10000000),
                timestamp=datetime.utcnow()
            )
            
            # Create mock order book
            self.order_books[symbol] = self._generate_order_book(symbol, current_price)
    
    def _generate_order_book(self, symbol: str, mid_price: float) -> OrderBook:
        """Generate realistic order book."""
        bids = []
        asks = []
        
        # Generate 10 levels on each side
        for i in range(10):
            # Bid side (buy orders)
            bid_price = mid_price * (1 - (i + 1) * 0.001)  # 0.1% apart
            bid_qty = random.uniform(0.1, 10.0)
            bids.append(OrderBookLevel(bid_price, bid_qty))
            
            # Ask side (sell orders) 
            ask_price = mid_price * (1 + (i + 1) * 0.001)
            ask_qty = random.uniform(0.1, 10.0)
            asks.append(OrderBookLevel(ask_price, ask_qty))
        
        return OrderBook(
            symbol=symbol,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),  # Highest first
            asks=sorted(asks, key=lambda x: x.price),  # Lowest first
            timestamp=datetime.utcnow()
        )
    
    def update_market_data(self, symbol: str, price: float, volume: float = None):
        """Update market data for a symbol."""
        if symbol not in self.market_data:
            self.market_data[symbol] = MarketData(
                symbol=symbol,
                bid=price * 0.9995,
                ask=price * 1.0005,
                last=price,
                volume_24h=volume or 1000000,
                timestamp=datetime.utcnow()
            )
        else:
            market_data = self.market_data[symbol]
            market_data.last = price
            market_data.bid = price * 0.9995
            market_data.ask = price * 1.0005
            market_data.timestamp = datetime.utcnow()
            if volume:
                market_data.volume_24h = volume
        
        # Update order book
        self.order_books[symbol] = self._generate_order_book(symbol, price)
    
    async def place_order(self, 
                         symbol: str,
                         side: OrderSide, 
                         quantity: float,
                         order_type: OrderType,
                         price: Optional[float] = None,
                         client_order_id: Optional[str] = None) -> Order:
        """Place a new order."""
        
        order_id = client_order_id or str(uuid.uuid4())
        
        # Create order object
        order = Order(
            id=order_id,
            exchange_id=1,  # Default exchange
            symbol=symbol,
            side=side,
            qty=Decimal(str(quantity)),
            type=order_type,
            price=Decimal(str(price)) if price else None,
            status=OrderStatus.PENDING,
            filled_qty=Decimal('0'),
            avg_fill_price=None,
            client_order_id=client_order_id,
            exchange_order_id=order_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store order
        self.orders[order_id] = order
        
        # Emit order event
        await self.event_bus.publish("orders", {
            "type": "order_placed",
            "order": order.dict() if hasattr(order, 'dict') else {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": float(order.qty),
                "type": order.type,
                "status": order.status
            }
        })
        
        # Process order asynchronously
        asyncio.create_task(self._process_order(order_id))
        
        return order
    
    async def _process_order(self, order_id: str):
        """Process order execution asynchronously."""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        try:
            # Simulate network latency
            latency_ms = random.uniform(*self.latency_range)
            await asyncio.sleep(latency_ms / 1000)
            
            # Check if order was canceled
            if order.status == OrderStatus.CANCELED:
                return
            
            # Update order status to open
            order.status = OrderStatus.OPEN
            order.updated_at = datetime.utcnow()
            
            # Execute order based on type
            if order.type == OrderType.MARKET:
                await self._execute_market_order(order)
            elif order.type == OrderType.LIMIT:
                await self._execute_limit_order(order)
                
        except Exception as e:
            logger.error(f"Error processing order {order_id}: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.utcnow()
        
        # Update stored order
        self.orders[order_id] = order
    
    async def _execute_market_order(self, order: Order):
        """Execute market order immediately."""
        symbol = order.symbol
        
        if symbol not in self.market_data:
            order.status = OrderStatus.REJECTED
            return
        
        market_data = self.market_data[symbol]
        
        # Determine execution price based on side
        if order.side == OrderSide.BUY:
            base_price = market_data.ask
        else:
            base_price = market_data.bid
        
        # Apply slippage based on order size and market conditions
        slippage = self._calculate_slippage(symbol, float(order.qty), base_price)
        
        if order.side == OrderSide.BUY:
            execution_price = base_price * (1 + slippage)
        else:
            execution_price = base_price * (1 - slippage)
        
        # Create fill
        await self._create_fill(order, execution_price, float(order.qty))
    
    async def _execute_limit_order(self, order: Order):
        """Execute limit order if price conditions are met."""
        symbol = order.symbol
        
        if symbol not in self.market_data:
            order.status = OrderStatus.REJECTED
            return
        
        market_data = self.market_data[symbol]
        limit_price = float(order.price)
        
        # Check if order can be filled immediately
        can_fill = False
        
        if order.side == OrderSide.BUY and market_data.ask <= limit_price:
            can_fill = True
            execution_price = min(market_data.ask, limit_price)
        elif order.side == OrderSide.SELL and market_data.bid >= limit_price:
            can_fill = True
            execution_price = max(market_data.bid, limit_price)
        
        if can_fill:
            await self._create_fill(order, execution_price, float(order.qty))
        # Otherwise, order remains open (limit orders stay in book)
    
    def _calculate_slippage(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate realistic slippage based on order size and market conditions."""
        if symbol not in self.market_data:
            return self.slippage_factor
        
        market_data = self.market_data[symbol]
        
        # Base slippage
        slippage = self.slippage_factor
        
        # Increase slippage for larger orders (market impact)
        order_value = quantity * price
        volume_impact = order_value / market_data.volume_24h
        
        # Scale slippage based on order size relative to daily volume
        if volume_impact > 0.001:  # > 0.1% of daily volume
            slippage *= (1 + volume_impact * 10)
        
        # Add some randomness to simulate market conditions
        slippage *= random.uniform(0.5, 1.5)
        
        # Cap slippage at 0.5%
        return min(slippage, 0.005)
    
    async def _create_fill(self, order: Order, price: float, quantity: float):
        """Create a fill for an order."""
        # Calculate fee
        trade_value = price * quantity
        fee = trade_value * self.base_fee_rate
        
        # Create fill record
        fill = Fill(
            id=str(uuid.uuid4()),
            order_id=order.id,
            price=Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            qty=Decimal(str(quantity)),
            fee=Decimal(str(fee)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            ts=datetime.utcnow(),
            trade_id=str(uuid.uuid4())
        )
        
        self.fills.append(fill)
        
        # Update order
        order.filled_qty = Decimal(str(quantity))
        order.avg_fill_price = Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.utcnow()
        
        # Update positions
        await self._update_position(order.symbol, order.side, quantity, price, fee)
        
        # Emit fill event
        await self.event_bus.publish("fills", {
            "type": "order_filled",
            "fill": {
                "id": fill.id,
                "order_id": fill.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "timestamp": fill.ts.isoformat()
            }
        })
        
        logger.info(f"Order {order.id} filled: {quantity} {order.symbol} at {price}")
    
    async def _update_position(self, symbol: str, side: OrderSide, quantity: float, price: float, fee: float):
        """Update position after a fill."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0.0,
                "avg_price": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_fees": 0.0
            }
        
        position = self.positions[symbol]
        current_qty = position["quantity"]
        current_avg_price = position["avg_price"]
        
        if side == OrderSide.BUY:
            # Adding to long position
            if current_qty >= 0:
                # Average up the position
                total_cost = (current_qty * current_avg_price) + (quantity * price)
                new_qty = current_qty + quantity
                new_avg_price = total_cost / new_qty if new_qty > 0 else 0
                
                position["quantity"] = new_qty
                position["avg_price"] = new_avg_price
            else:
                # Reducing short position
                if abs(current_qty) >= quantity:
                    # Partial close of short
                    realized_pnl = quantity * (current_avg_price - price)
                    position["quantity"] = current_qty + quantity
                    position["realized_pnl"] += realized_pnl
                else:
                    # Close short and go long
                    close_qty = abs(current_qty)
                    open_qty = quantity - close_qty
                    
                    realized_pnl = close_qty * (current_avg_price - price)
                    position["realized_pnl"] += realized_pnl
                    position["quantity"] = open_qty
                    position["avg_price"] = price
        
        else:  # SELL
            # Adding to short position or reducing long
            if current_qty <= 0:
                # Average down the short position
                total_cost = (abs(current_qty) * current_avg_price) + (quantity * price)
                new_qty = current_qty - quantity  # More negative
                new_avg_price = total_cost / abs(new_qty) if new_qty != 0 else 0
                
                position["quantity"] = new_qty
                position["avg_price"] = new_avg_price
            else:
                # Reducing long position
                if current_qty >= quantity:
                    # Partial close of long
                    realized_pnl = quantity * (price - current_avg_price)
                    position["quantity"] = current_qty - quantity
                    position["realized_pnl"] += realized_pnl
                else:
                    # Close long and go short
                    close_qty = current_qty
                    open_qty = quantity - close_qty
                    
                    realized_pnl = close_qty * (price - current_avg_price)
                    position["realized_pnl"] += realized_pnl
                    position["quantity"] = -open_qty  # Negative for short
                    position["avg_price"] = price
        
        position["total_fees"] += fee
        
        # Calculate unrealized PnL
        if symbol in self.market_data:
            current_price = self.market_data[symbol].last
            if position["quantity"] > 0:
                position["unrealized_pnl"] = position["quantity"] * (current_price - position["avg_price"])
            elif position["quantity"] < 0:
                position["unrealized_pnl"] = abs(position["quantity"]) * (position["avg_price"] - current_price)
            else:
                position["unrealized_pnl"] = 0.0
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
            return False
        
        order.status = OrderStatus.CANCELED
        order.updated_at = datetime.utcnow()
        
        await self.event_bus.publish("orders", {
            "type": "order_canceled",
            "order_id": order_id,
            "symbol": order.symbol
        })
        
        return True
    
    def get_orders(self, symbol: str = None, status: OrderStatus = None) -> List[Order]:
        """Get orders with optional filtering."""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return sorted(orders, key=lambda x: x.created_at, reverse=True)
    
    def get_fills(self, symbol: str = None) -> List[Fill]:
        """Get fills with optional filtering."""
        fills = self.fills.copy()
        
        if symbol:
            fills = [f for f in fills if f.order_id in self.orders and self.orders[f.order_id].symbol == symbol]
        
        return sorted(fills, key=lambda x: x.ts, reverse=True)
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        positions = []
        
        for symbol, position in self.positions.items():
            if abs(position["quantity"]) > 0.001:  # Filter out tiny positions
                pos_data = position.copy()
                
                # Add current market value
                if symbol in self.market_data:
                    current_price = self.market_data[symbol].last
                    pos_data["current_price"] = current_price
                    pos_data["market_value"] = abs(position["quantity"]) * current_price
                    
                    # Recalculate unrealized PnL
                    if position["quantity"] > 0:
                        pos_data["unrealized_pnl"] = position["quantity"] * (current_price - position["avg_price"])
                    elif position["quantity"] < 0:
                        pos_data["unrealized_pnl"] = abs(position["quantity"]) * (position["avg_price"] - current_price)
                
                positions.append(pos_data)
        
        return positions
    
    def get_balance(self, currency: str = "USDT") -> float:
        """Get account balance for a currency."""
        # Start with initial balance
        balance = 10000.0  # Initial paper trading balance
        
        # Subtract realized losses and add realized gains
        for position in self.positions.values():
            balance += position["realized_pnl"]
            balance -= position["total_fees"]
        
        # For USDT, subtract value of open positions
        if currency == "USDT":
            for position in self.positions.values():
                if position["quantity"] != 0 and position["symbol"].endswith("/USDT"):
                    # This is the cash tied up in positions
                    balance -= abs(position["quantity"]) * position["avg_price"]
        
        return max(balance, 0.0)  # Can't go negative in paper trading
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        cash_balance = self.get_balance("USDT")
        positions_value = 0.0
        
        for position in self.get_positions():
            positions_value += position.get("market_value", 0.0)
        
        return cash_balance + positions_value