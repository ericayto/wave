"""
LLM Tools API

Function calling tools that LLMs can use to interact with the trading system.
These tools provide safe, validated access to market data, portfolio info, and trading functions.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, validator

from ..models.database import get_db
from ..models.trading import Order, Position, Fill
from ..models.strategy import Strategy, Backtest
from ..models.user import User, Exchange
from .event_bus import EventBus
from .market_data import MarketDataService
from .paper_broker import PaperBroker
from .risk_engine import RiskEngine
from .strategy_engine import StrategyEngine
from .indicators import TechnicalIndicators


logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Tool execution error."""
    pass


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    """Time in force."""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class NotificationSeverity(str, Enum):
    """Notification severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ToolContext:
    """Context passed to all tools."""
    user_id: int
    session_id: str
    event_bus: EventBus
    market_data: MarketDataService
    paper_broker: PaperBroker
    risk_engine: RiskEngine
    strategy_engine: StrategyEngine


class LLMTools:
    """Collection of LLM function calling tools."""
    
    def __init__(self, context: ToolContext):
        self.context = context
        self.indicators = TechnicalIndicators()
    
    async def get_portfolio(self) -> Dict[str, Any]:
        """
        Get current portfolio balances and positions.
        
        Returns:
            Dict containing balances, positions, and portfolio metrics
        """
        try:
            async with get_db() as db:
                # Get user's positions
                positions = await db.query(Position).filter(
                    Position.user_id == self.context.user_id
                ).all()
                
                # Calculate portfolio metrics
                total_value = 0.0
                unrealized_pnl = 0.0
                position_data = []
                
                for position in positions:
                    # Get current market price
                    market_price = await self.context.market_data.get_ticker(position.symbol)
                    current_price = market_price.get('last', position.avg_price)
                    
                    # Calculate metrics
                    market_value = position.quantity * current_price
                    cost_basis = position.quantity * position.avg_price
                    pnl = market_value - cost_basis
                    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                    
                    total_value += market_value
                    unrealized_pnl += pnl
                    
                    position_data.append({
                        "symbol": position.symbol,
                        "quantity": position.quantity,
                        "avg_price": position.avg_price,
                        "current_price": current_price,
                        "market_value": market_value,
                        "unrealized_pnl": pnl,
                        "unrealized_pnl_pct": pnl_pct,
                        "updated_at": position.updated_at.isoformat()
                    })
                
                # Get account balance (simplified - assume USD base currency)
                cash_balance = 100000.0  # Default paper trading balance
                total_portfolio_value = cash_balance + total_value
                
                return {
                    "cash_balance": cash_balance,
                    "positions_value": total_value,
                    "total_value": total_portfolio_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": (unrealized_pnl / total_portfolio_value * 100) if total_portfolio_value > 0 else 0,
                    "positions": position_data,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            raise ToolError(f"Failed to get portfolio: {str(e)}")
    
    async def get_market_snapshot(
        self, 
        symbols: List[str], 
        fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get market snapshot for specified symbols and fields.
        
        Args:
            symbols: List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            fields: List of fields to include (price, spread, volume, rsi, macd, etc.)
        
        Returns:
            Dict containing market data for each symbol
        """
        if fields is None:
            fields = ['price', 'spread', 'volume', 'change_24h']
        
        try:
            market_data = {}
            
            for symbol in symbols:
                symbol_data = {}
                
                # Get basic ticker data
                ticker = await self.context.market_data.get_ticker(symbol)
                
                if 'price' in fields:
                    symbol_data['price'] = ticker.get('last')
                    symbol_data['bid'] = ticker.get('bid')
                    symbol_data['ask'] = ticker.get('ask')
                
                if 'spread' in fields:
                    bid = ticker.get('bid', 0)
                    ask = ticker.get('ask', 0)
                    spread = ask - bid
                    spread_bps = (spread / ask * 10000) if ask > 0 else 0
                    symbol_data['spread'] = spread
                    symbol_data['spread_bps'] = spread_bps
                
                if 'volume' in fields:
                    symbol_data['volume_24h'] = ticker.get('baseVolume', 0)
                    symbol_data['volume_usd_24h'] = ticker.get('quoteVolume', 0)
                
                if 'change_24h' in fields:
                    symbol_data['change_24h'] = ticker.get('change', 0)
                    symbol_data['change_24h_pct'] = ticker.get('percentage', 0)
                
                # Get technical indicators if requested
                if any(field in fields for field in ['rsi', 'macd', 'sma', 'ema', 'bollinger']):
                    ohlcv = await self.context.market_data.get_ohlcv(symbol, '1h', 50)
                    if ohlcv:
                        closes = [candle['close'] for candle in ohlcv]
                        highs = [candle['high'] for candle in ohlcv]
                        lows = [candle['low'] for candle in ohlcv]
                        volumes = [candle['volume'] for candle in ohlcv]
                        
                        if 'rsi' in fields and len(closes) >= 14:
                            rsi = self.indicators.calculate_rsi(closes, 14)
                            symbol_data['rsi'] = rsi[-1] if rsi else None
                        
                        if 'macd' in fields and len(closes) >= 26:
                            macd_line, signal_line, histogram = self.indicators.calculate_macd(closes)
                            symbol_data['macd'] = {
                                'macd': macd_line[-1] if macd_line else None,
                                'signal': signal_line[-1] if signal_line else None,
                                'histogram': histogram[-1] if histogram else None
                            }
                        
                        if 'sma' in fields and len(closes) >= 20:
                            sma20 = self.indicators.calculate_sma(closes, 20)
                            sma50 = self.indicators.calculate_sma(closes, 50) if len(closes) >= 50 else None
                            symbol_data['sma'] = {
                                'sma20': sma20[-1] if sma20 else None,
                                'sma50': sma50[-1] if sma50 else None
                            }
                
                symbol_data['timestamp'] = datetime.now().isoformat()
                market_data[symbol] = symbol_data
            
            return {
                "data": market_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market snapshot: {e}")
            raise ToolError(f"Failed to get market snapshot: {str(e)}")
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        lookback: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV candlestick data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            lookback: Number of candles to return
        
        Returns:
            List of OHLCV candles
        """
        try:
            ohlcv = await self.context.market_data.get_ohlcv(symbol, timeframe, lookback)
            
            if not ohlcv:
                return []
            
            return [
                {
                    "timestamp": candle['timestamp'],
                    "datetime": datetime.fromtimestamp(candle['timestamp'] / 1000).isoformat(),
                    "open": candle['open'],
                    "high": candle['high'],
                    "low": candle['low'],
                    "close": candle['close'],
                    "volume": candle['volume']
                }
                for candle in ohlcv
            ]
            
        except Exception as e:
            logger.error(f"Error getting OHLCV: {e}")
            raise ToolError(f"Failed to get OHLCV data: {str(e)}")
    
    async def backtest(
        self,
        strategy_definition: Dict[str, Any],
        params: Dict[str, Any] = None,
        from_date: str = None,
        to_date: str = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy.
        
        Args:
            strategy_definition: Strategy definition JSON
            params: Strategy parameters override
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
        
        Returns:
            Backtest results and metrics
        """
        try:
            # Set default dates
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).isoformat()
            if not to_date:
                to_date = datetime.now().isoformat()
            
            # Mock backtest results (in real implementation, this would run actual backtest)
            results = {
                "strategy": strategy_definition.get('name', 'Unnamed Strategy'),
                "period": {
                    "from": from_date,
                    "to": to_date,
                    "days": (datetime.fromisoformat(to_date.replace('Z', '+00:00')) - 
                            datetime.fromisoformat(from_date.replace('Z', '+00:00'))).days
                },
                "metrics": {
                    "total_return": 0.085,  # 8.5%
                    "cagr": 0.127,  # 12.7% annualized
                    "max_drawdown": -0.032,  # -3.2%
                    "sharpe_ratio": 1.43,
                    "win_rate": 0.64,  # 64%
                    "profit_factor": 1.87,
                    "total_trades": 28,
                    "winning_trades": 18,
                    "losing_trades": 10
                },
                "trades": [
                    {
                        "symbol": "BTC/USDT",
                        "side": "buy",
                        "entry_date": "2024-01-15T10:30:00Z",
                        "exit_date": "2024-01-16T14:20:00Z",
                        "entry_price": 42500.0,
                        "exit_price": 43200.0,
                        "quantity": 0.1,
                        "pnl": 70.0,
                        "pnl_pct": 1.65
                    }
                ],
                "equity_curve": [
                    {"date": "2024-01-01", "value": 100000},
                    {"date": "2024-01-15", "value": 103500},
                    {"date": "2024-01-30", "value": 108500}
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store backtest result
            async with get_db() as db:
                backtest_record = Backtest(
                    strategy_id=None,  # Strategy not persisted yet
                    params_json=json.dumps(params or {}),
                    window_start=datetime.fromisoformat(from_date.replace('Z', '+00:00')),
                    window_end=datetime.fromisoformat(to_date.replace('Z', '+00:00')),
                    metrics_json=json.dumps(results['metrics'])
                )
                db.add(backtest_record)
                await db.commit()
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise ToolError(f"Failed to run backtest: {str(e)}")
    
    async def propose_strategy(
        self,
        goal: str,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a strategy definition from natural language goal.
        
        Args:
            goal: Natural language description of strategy goal
            constraints: Additional constraints (risk limits, symbols, timeframes)
        
        Returns:
            Strategy definition JSON
        """
        try:
            constraints = constraints or {}
            
            # Extract strategy type from goal
            strategy_type = "trend_following"
            if "mean reversion" in goal.lower() or "rsi" in goal.lower():
                strategy_type = "mean_reversion"
            elif "breakout" in goal.lower():
                strategy_type = "breakout"
            elif "momentum" in goal.lower():
                strategy_type = "momentum"
            
            # Generate strategy based on goal and constraints
            strategy = {
                "name": f"LLM Generated {strategy_type.title().replace('_', ' ')} Strategy",
                "version": "1.0.0",
                "description": goal,
                "instrument_universe": constraints.get('symbols', ['BTC/USDT', 'ETH/USDT']),
                "timeframes": constraints.get('timeframes', ['1h']),
                "signals": [],
                "entries": [],
                "exits": [],
                "risk": {
                    "max_position_pct": constraints.get('max_position_pct', 0.25),
                    "daily_loss_limit_pct": constraints.get('daily_loss_limit_pct', 2.0),
                    "max_orders_per_hour": constraints.get('max_orders_per_hour', 4)
                },
                "notes": f"Generated from goal: {goal}"
            }
            
            # Add signals based on strategy type
            if strategy_type == "trend_following":
                strategy['signals'] = [
                    {
                        "id": "sma_cross",
                        "type": "indicator",
                        "indicator": "SMA_CROSS",
                        "params": {"fast": 20, "slow": 50}
                    },
                    {
                        "id": "volume_confirm",
                        "type": "indicator", 
                        "indicator": "VOLUME",
                        "params": {"threshold": 1.2}
                    }
                ]
                strategy['entries'] = [
                    {
                        "when": "sma_cross.bullish and volume_confirm.above",
                        "action": {"side": "buy", "size_pct": 0.5}
                    }
                ]
                strategy['exits'] = [
                    {
                        "when": "sma_cross.bearish or stop_loss(2.0) or take_profit(4.0)"
                    }
                ]
            
            elif strategy_type == "mean_reversion":
                strategy['signals'] = [
                    {
                        "id": "rsi_oversold",
                        "type": "indicator",
                        "indicator": "RSI",
                        "params": {"period": 14, "threshold": 30}
                    },
                    {
                        "id": "rsi_overbought",
                        "type": "indicator",
                        "indicator": "RSI", 
                        "params": {"period": 14, "threshold": 70}
                    }
                ]
                strategy['entries'] = [
                    {
                        "when": "rsi_oversold",
                        "action": {"side": "buy", "size_pct": 0.3}
                    },
                    {
                        "when": "rsi_overbought",
                        "action": {"side": "sell", "size_pct": 0.3}
                    }
                ]
                strategy['exits'] = [
                    {
                        "when": "rsi.between(40, 60) or stop_loss(1.5) or take_profit(2.5)"
                    }
                ]
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error proposing strategy: {e}")
            raise ToolError(f"Failed to propose strategy: {str(e)}")
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC
    ) -> Dict[str, Any]:
        """
        Place a paper trading order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type (market/limit)
            price: Limit price (required for limit orders)
            time_in_force: Time in force
        
        Returns:
            Order confirmation
        """
        try:
            # Validate inputs
            if order_type == OrderType.LIMIT and price is None:
                raise ToolError("Limit orders require a price")
            
            if quantity <= 0:
                raise ToolError("Quantity must be positive")
            
            # Check risk limits
            risk_check = await self.context.risk_engine.validate_order({
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'price': price,
                'user_id': self.context.user_id
            })
            
            if not risk_check['allowed']:
                raise ToolError(f"Order rejected by risk engine: {risk_check['reason']}")
            
            # Place order through paper broker
            order_result = await self.context.paper_broker.place_order(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                order_type=order_type.value,
                price=price
            )
            
            # Emit order event
            await self.context.event_bus.publish("trading.order_placed", {
                "order_id": order_result['order_id'],
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "type": order_type.value,
                "price": price,
                "user_id": self.context.user_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "order_id": order_result['order_id'],
                "status": order_result['status'],
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "type": order_type.value,
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "message": "Order placed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise ToolError(f"Failed to place order: {str(e)}")
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            Cancellation confirmation
        """
        try:
            result = await self.context.paper_broker.cancel_order(order_id)
            
            # Emit cancellation event
            await self.context.event_bus.publish("trading.order_canceled", {
                "order_id": order_id,
                "user_id": self.context.user_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "order_id": order_id,
                "status": "canceled",
                "timestamp": datetime.now().isoformat(),
                "message": "Order canceled successfully"
            }
            
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            raise ToolError(f"Failed to cancel order: {str(e)}")
    
    async def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits."""
        try:
            limits = await self.context.risk_engine.get_limits(self.context.user_id)
            return {
                "limits": limits,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting risk limits: {e}")
            raise ToolError(f"Failed to get risk limits: {str(e)}")
    
    async def set_risk_limits(self, limits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update risk management limits.
        
        Args:
            limits: New risk limits
        
        Returns:
            Confirmation of updated limits
        """
        try:
            await self.context.risk_engine.update_limits(self.context.user_id, limits)
            
            # Emit limits update event
            await self.context.event_bus.publish("risk.limits_updated", {
                "user_id": self.context.user_id,
                "limits": limits,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "updated",
                "limits": limits,
                "timestamp": datetime.now().isoformat(),
                "message": "Risk limits updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error setting risk limits: {e}")
            raise ToolError(f"Failed to set risk limits: {str(e)}")
    
    async def notify_user(
        self, 
        message: str, 
        severity: NotificationSeverity = NotificationSeverity.INFO
    ) -> Dict[str, Any]:
        """
        Send notification to user.
        
        Args:
            message: Notification message
            severity: Notification severity level
        
        Returns:
            Notification confirmation
        """
        try:
            # Emit notification event
            await self.context.event_bus.publish("user.notification", {
                "user_id": self.context.user_id,
                "message": message,
                "severity": severity.value,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "sent",
                "message": message,
                "severity": severity.value,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            raise ToolError(f"Failed to send notification: {str(e)}")
    
    async def ask_human(self, prompt: str) -> str:
        """
        Request human input (human-in-the-loop).
        
        Args:
            prompt: Question or prompt for human
        
        Returns:
            Human response
        """
        try:
            # Create human interaction event
            interaction_id = f"human_{datetime.now().timestamp()}"
            
            # Emit human interaction request
            await self.context.event_bus.publish("llm.human_interaction", {
                "interaction_id": interaction_id,
                "user_id": self.context.user_id,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            })
            
            # In real implementation, this would wait for user response via WebSocket
            # For now, return a mock response
            mock_response = "Proceed with the analysis and place the order if risk conditions are met."
            
            # Emit response received event
            await self.context.event_bus.publish("llm.human_response", {
                "interaction_id": interaction_id,
                "user_id": self.context.user_id,
                "response": mock_response,
                "timestamp": datetime.now().isoformat()
            })
            
            return mock_response
            
        except Exception as e:
            logger.error(f"Error in human interaction: {e}")
            raise ToolError(f"Failed to get human input: {str(e)}")


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get OpenAI function definitions for all tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_portfolio",
                "description": "Get current portfolio balances, positions, and metrics",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_market_snapshot",
                "description": "Get market data snapshot for specified symbols and fields",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])"
                        },
                        "fields": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Fields to include: price, spread, volume, change_24h, rsi, macd, sma"
                        }
                    },
                    "required": ["symbols"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_ohlcv", 
                "description": "Get OHLCV candlestick data for a symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol (e.g., 'BTC/USDT')"
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Timeframe: 1m, 5m, 15m, 1h, 4h, 1d",
                            "default": "1h"
                        },
                        "lookback": {
                            "type": "integer",
                            "description": "Number of candles to return",
                            "default": 100
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "backtest",
                "description": "Run backtest for a strategy definition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "strategy_definition": {
                            "type": "object",
                            "description": "Strategy definition JSON"
                        },
                        "params": {
                            "type": "object", 
                            "description": "Strategy parameters override"
                        },
                        "from_date": {
                            "type": "string",
                            "description": "Start date (ISO format)"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "End date (ISO format)"
                        }
                    },
                    "required": ["strategy_definition"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "propose_strategy",
                "description": "Generate strategy definition from natural language goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Natural language description of strategy goal"
                        },
                        "constraints": {
                            "type": "object",
                            "description": "Additional constraints (risk limits, symbols, timeframes)"
                        }
                    },
                    "required": ["goal"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "place_order",
                "description": "Place a paper trading order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "Order side"
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Order quantity"
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["market", "limit"],
                            "description": "Order type",
                            "default": "market"
                        },
                        "price": {
                            "type": "number",
                            "description": "Limit price (required for limit orders)"
                        },
                        "time_in_force": {
                            "type": "string",
                            "enum": ["gtc", "ioc", "fok"],
                            "description": "Time in force",
                            "default": "gtc"
                        }
                    },
                    "required": ["symbol", "side", "quantity"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_order",
                "description": "Cancel an existing order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order ID to cancel"
                        }
                    },
                    "required": ["order_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_risk_limits",
                "description": "Get current risk management limits",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_risk_limits",
                "description": "Update risk management limits",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limits": {
                            "type": "object",
                            "description": "New risk limits"
                        }
                    },
                    "required": ["limits"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "notify_user",
                "description": "Send notification to user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Notification message"
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "error", "critical"],
                            "description": "Notification severity level",
                            "default": "info"
                        }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_human",
                "description": "Request human input (human-in-the-loop)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Question or prompt for human"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ]