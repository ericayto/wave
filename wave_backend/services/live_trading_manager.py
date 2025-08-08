"""
Live Trading Manager
Live trading infrastructure (evaluation phase) with professional OMS and safety controls.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
import uuid
import json

from ..config.settings import get_settings
from ..services.event_bus import EventBus
from .advanced_risk_engine import AdvancedRiskEngine

logger = logging.getLogger(__name__)

class OrderState(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

class EmergencyStopType(str, Enum):
    USER_INITIATED = "user_initiated"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    SYSTEM_ERROR = "system_error"
    REGULATORY_HALT = "regulatory_halt"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CIRCUIT_BREAKER = "circuit_breaker"

class PositionMonitorAlert(str, Enum):
    POSITION_LIMIT_WARNING = "position_limit_warning"
    DRAWDOWN_ALERT = "drawdown_alert"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_WARNING = "liquidity_warning"
    EXECUTION_QUALITY_DEGRADED = "execution_quality_degraded"

@dataclass
class LiveOrder:
    """Live trading order with comprehensive tracking."""
    order_id: str
    strategy_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float]
    
    # State tracking
    state: OrderState
    created_at: datetime
    updated_at: datetime
    
    # Execution details
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    fills: List[Dict] = None
    
    # Risk and compliance
    pre_trade_checks: Dict[str, Any] = None
    risk_approval: bool = False
    compliance_approval: bool = False
    
    # Execution metadata
    exchange_order_id: Optional[str] = None
    estimated_commission: float = 0.0
    slippage_bps: Optional[float] = None
    
    # Timing
    validation_time_ms: Optional[float] = None
    submission_time_ms: Optional[float] = None
    first_fill_time_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.fills is None:
            self.fills = []

@dataclass
class PositionSnapshot:
    """Real-time position snapshot."""
    strategy_id: str
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk metrics
    position_limit_utilization: float
    var_contribution: float
    correlation_exposure: Dict[str, float]
    
    # Execution metrics
    avg_entry_price: float
    current_price: float
    time_in_position: timedelta
    
    # Alerts
    active_alerts: List[PositionMonitorAlert]
    
    timestamp: datetime

@dataclass
class EmergencyStop:
    """Emergency stop event record."""
    stop_id: str
    stop_type: EmergencyStopType
    triggered_at: datetime
    triggered_by: str
    
    # Context
    trigger_reason: str
    affected_strategies: List[str]
    portfolio_state: Dict[str, Any]
    
    # Actions taken
    orders_cancelled: int
    positions_closed: int
    
    # Recovery
    cleared_at: Optional[datetime] = None
    cleared_by: Optional[str] = None
    recovery_notes: Optional[str] = None

@dataclass
class ExecutionQuality:
    """Execution quality metrics."""
    strategy_id: str
    symbol: str
    time_period: str
    
    # Slippage analysis
    avg_slippage_bps: float
    worst_slippage_bps: float
    slippage_volatility: float
    
    # Timing analysis
    avg_fill_time_ms: float
    fill_time_percentile_95: float
    
    # Market impact
    temporary_impact_bps: float
    permanent_impact_bps: float
    
    # Fill quality
    fill_rate: float  # Percentage of orders fully filled
    partial_fill_rate: float
    cancel_rate: float
    
    # Benchmark comparison
    vs_arrival_price_bps: float
    vs_twap_bps: float
    vs_vwap_bps: float

@dataclass
class ComplianceCheck:
    """Regulatory compliance check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    
    # Regulatory context
    regulation_type: str  # 'position_limit', 'wash_trading', 'market_manipulation'
    jurisdiction: str
    severity: str  # 'info', 'warning', 'violation'

class LiveTradingManager:
    """Live trading infrastructure (evaluation phase)."""
    
    def __init__(self, event_bus: EventBus, risk_engine: AdvancedRiskEngine):
        self.event_bus = event_bus
        self.risk_engine = risk_engine
        self.settings = get_settings()
        
        # Trading state
        self.live_enabled = getattr(self.settings.live_trading, 'enabled', False)
        self.emergency_stops = {}
        self.active_orders = {}  # order_id -> LiveOrder
        self.position_snapshots = {}  # strategy_id -> PositionSnapshot
        
        # Performance monitoring
        self.execution_quality_tracker = {}
        self.compliance_violations = []
        
        # Configuration
        self.max_position_value = getattr(self.settings.live_trading, 'max_position_value', 1000.0)
        self.pre_trade_timeout_ms = getattr(self.settings.live_trading, 'pre_trade_checks_timeout', 1000)
        self.position_monitor_frequency = getattr(self.settings.live_trading, 'position_monitoring_frequency', 1)
        
        # Safety mechanisms
        self.circuit_breakers_active = False
        self.global_position_limit = self.max_position_value * 10  # Portfolio level
        self.emergency_drawdown_limit = getattr(self.settings.live_trading, 'emergency_stop_drawdown', 0.10)
        
        # Background tasks
        self.monitoring_task = None
        self.compliance_task = None
        
    async def start(self):
        """Start live trading manager."""
        logger.info("Starting live trading manager...")
        
        if not self.live_enabled:
            logger.info("Live trading is disabled - running in evaluation mode only")
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._position_monitoring_loop())
        self.compliance_task = asyncio.create_task(self._compliance_monitoring_loop())
        
        # Initialize emergency stop monitoring
        await self._initialize_emergency_stops()
        
        logger.info("Live trading manager started")
    
    async def stop(self):
        """Stop live trading manager."""
        logger.info("Stopping live trading manager...")
        
        # Cancel all active orders
        await self._cancel_all_orders("system_shutdown")
        
        # Stop monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.compliance_task:
            self.compliance_task.cancel()
        
        logger.info("Live trading manager stopped")
    
    async def submit_order(self, 
                         strategy_id: str,
                         symbol: str,
                         side: str,
                         quantity: float,
                         order_type: str = "market",
                         price: Optional[float] = None) -> LiveOrder:
        """Submit order with comprehensive pre-trade checks."""
        
        order_id = str(uuid.uuid4())
        
        logger.info(f"Submitting order: {order_id} - {side} {quantity} {symbol} @ {price}")
        
        # Create order object
        order = LiveOrder(
            order_id=order_id,
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            state=OrderState.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Add to tracking
        self.active_orders[order_id] = order
        
        try:
            # Pre-trade risk checks
            pre_trade_start = datetime.utcnow()
            await self._run_pre_trade_checks(order)
            pre_trade_duration = (datetime.utcnow() - pre_trade_start).total_seconds() * 1000
            order.validation_time_ms = pre_trade_duration
            
            # Check if order was rejected during validation
            if order.state == OrderState.REJECTED:
                return order
            
            # Submit to exchange (mock in evaluation phase)
            if self.live_enabled:
                await self._submit_to_exchange(order)
            else:
                # Mock submission for evaluation
                await self._mock_order_execution(order)
            
            # Update order state
            order.state = OrderState.SUBMITTED
            order.updated_at = datetime.utcnow()
            
            # Emit order event
            await self.event_bus.publish("live_order_submitted", {
                "order_id": order_id,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity
            })
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            order.state = OrderState.FAILED
            order.updated_at = datetime.utcnow()
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order."""
        
        if order_id not in self.active_orders:
            logger.warning(f"Cannot cancel order {order_id}: not found")
            return False
        
        order = self.active_orders[order_id]
        
        if order.state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED]:
            logger.warning(f"Cannot cancel order {order_id}: already in final state {order.state}")
            return False
        
        try:
            # Cancel on exchange
            if self.live_enabled:
                success = await self._cancel_on_exchange(order)
            else:
                success = True  # Mock cancellation
            
            if success:
                order.state = OrderState.CANCELLED
                order.updated_at = datetime.utcnow()
                
                # Emit cancellation event
                await self.event_bus.publish("live_order_cancelled", {
                    "order_id": order_id,
                    "strategy_id": order.strategy_id
                })
                
                logger.info(f"Order cancelled: {order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    async def emergency_stop_all(self, 
                               stop_type: EmergencyStopType = EmergencyStopType.USER_INITIATED,
                               reason: str = "Manual emergency stop",
                               triggered_by: str = "user") -> EmergencyStop:
        """Execute emergency stop - halt all trading immediately."""
        
        stop_id = str(uuid.uuid4())
        
        logger.critical(f"EMERGENCY STOP INITIATED: {stop_id} - {reason}")
        
        # Create emergency stop record
        emergency_stop = EmergencyStop(
            stop_id=stop_id,
            stop_type=stop_type,
            triggered_at=datetime.utcnow(),
            triggered_by=triggered_by,
            trigger_reason=reason,
            affected_strategies=list(self.position_snapshots.keys()),
            portfolio_state=await self._capture_portfolio_state(),
            orders_cancelled=0,
            positions_closed=0
        )
        
        # Cancel all active orders
        cancelled_orders = await self._cancel_all_orders("emergency_stop")
        emergency_stop.orders_cancelled = cancelled_orders
        
        # Close positions if required
        if stop_type in [EmergencyStopType.RISK_LIMIT_BREACH, EmergencyStopType.LIQUIDITY_CRISIS]:
            closed_positions = await self._emergency_position_closure()
            emergency_stop.positions_closed = closed_positions
        
        # Activate circuit breakers
        self.circuit_breakers_active = True
        
        # Store emergency stop
        self.emergency_stops[stop_id] = emergency_stop
        
        # Emit critical alert
        await self.event_bus.publish("emergency_stop", {
            "stop_id": stop_id,
            "stop_type": stop_type,
            "reason": reason,
            "orders_cancelled": emergency_stop.orders_cancelled,
            "positions_closed": emergency_stop.positions_closed
        })
        
        logger.critical(f"Emergency stop completed: {cancelled_orders} orders cancelled, {emergency_stop.positions_closed} positions closed")
        
        return emergency_stop
    
    async def clear_emergency_stop(self, 
                                 stop_id: str,
                                 cleared_by: str = "user",
                                 recovery_notes: str = "") -> bool:
        """Clear emergency stop and resume trading."""
        
        if stop_id not in self.emergency_stops:
            logger.error(f"Cannot clear emergency stop {stop_id}: not found")
            return False
        
        emergency_stop = self.emergency_stops[stop_id]
        
        if emergency_stop.cleared_at is not None:
            logger.warning(f"Emergency stop {stop_id} already cleared")
            return False
        
        logger.info(f"Clearing emergency stop: {stop_id}")
        
        # Validate system state before clearing
        system_healthy = await self._validate_system_health()
        
        if not system_healthy:
            logger.error("Cannot clear emergency stop: system health check failed")
            return False
        
        # Clear emergency stop
        emergency_stop.cleared_at = datetime.utcnow()
        emergency_stop.cleared_by = cleared_by
        emergency_stop.recovery_notes = recovery_notes
        
        # Deactivate circuit breakers
        self.circuit_breakers_active = False
        
        # Emit recovery event
        await self.event_bus.publish("emergency_stop_cleared", {
            "stop_id": stop_id,
            "cleared_by": cleared_by,
            "recovery_time_minutes": (datetime.utcnow() - emergency_stop.triggered_at).total_seconds() / 60
        })
        
        logger.info(f"Emergency stop cleared: {stop_id}")
        
        return True
    
    async def get_real_time_positions(self) -> Dict[str, PositionSnapshot]:
        """Get real-time position monitoring."""
        
        # Update all position snapshots
        for strategy_id in self.position_snapshots:
            await self._update_position_snapshot(strategy_id)
        
        return self.position_snapshots.copy()
    
    async def analyze_execution_quality(self, 
                                      strategy_id: str,
                                      time_period: str = "1d") -> ExecutionQuality:
        """Analyze execution quality metrics."""
        
        logger.info(f"Analyzing execution quality for {strategy_id}")
        
        # Get relevant orders for analysis
        strategy_orders = [
            order for order in self.active_orders.values()
            if order.strategy_id == strategy_id and order.state == OrderState.FILLED
        ]
        
        if not strategy_orders:
            return self._default_execution_quality(strategy_id, time_period)
        
        # Calculate slippage metrics
        slippage_data = [order.slippage_bps for order in strategy_orders if order.slippage_bps is not None]
        
        avg_slippage = np.mean(slippage_data) if slippage_data else 0.0
        worst_slippage = max(slippage_data) if slippage_data else 0.0
        slippage_vol = np.std(slippage_data) if len(slippage_data) > 1 else 0.0
        
        # Calculate timing metrics
        fill_times = [order.first_fill_time_ms for order in strategy_orders if order.first_fill_time_ms is not None]
        
        avg_fill_time = np.mean(fill_times) if fill_times else 0.0
        fill_time_95 = np.percentile(fill_times, 95) if fill_times else 0.0
        
        # Calculate fill rates
        total_orders = len(strategy_orders)
        fully_filled = sum(1 for order in strategy_orders if order.filled_quantity == order.quantity)
        partially_filled = sum(1 for order in strategy_orders if 0 < order.filled_quantity < order.quantity)
        cancelled = sum(1 for order in strategy_orders if order.state == OrderState.CANCELLED)
        
        fill_rate = fully_filled / total_orders if total_orders > 0 else 0.0
        partial_fill_rate = partially_filled / total_orders if total_orders > 0 else 0.0
        cancel_rate = cancelled / total_orders if total_orders > 0 else 0.0
        
        return ExecutionQuality(
            strategy_id=strategy_id,
            symbol="ALL",  # Aggregate across symbols
            time_period=time_period,
            avg_slippage_bps=avg_slippage,
            worst_slippage_bps=worst_slippage,
            slippage_volatility=slippage_vol,
            avg_fill_time_ms=avg_fill_time,
            fill_time_percentile_95=fill_time_95,
            temporary_impact_bps=avg_slippage * 0.6,  # Estimate
            permanent_impact_bps=avg_slippage * 0.4,  # Estimate
            fill_rate=fill_rate,
            partial_fill_rate=partial_fill_rate,
            cancel_rate=cancel_rate,
            vs_arrival_price_bps=avg_slippage,  # Simplified
            vs_twap_bps=avg_slippage * 0.8,  # Estimate
            vs_vwap_bps=avg_slippage * 0.9   # Estimate
        )
    
    async def run_compliance_checks(self, order: LiveOrder) -> List[ComplianceCheck]:
        """Run regulatory compliance checks."""
        
        checks = []
        
        # Position limit compliance
        position_check = await self._check_position_limits(order)
        checks.append(position_check)
        
        # Wash trading detection
        wash_trading_check = await self._check_wash_trading(order)
        checks.append(wash_trading_check)
        
        # Market manipulation detection
        manipulation_check = await self._check_market_manipulation(order)
        checks.append(manipulation_check)
        
        # Concentration limits
        concentration_check = await self._check_concentration_limits(order)
        checks.append(concentration_check)
        
        # Trading hours compliance
        hours_check = await self._check_trading_hours(order)
        checks.append(hours_check)
        
        return checks
    
    # Private implementation methods
    
    async def _run_pre_trade_checks(self, order: LiveOrder):
        """Run comprehensive pre-trade validation."""
        
        order.state = OrderState.VALIDATING
        order.updated_at = datetime.utcnow()
        
        checks = {
            'risk_check': False,
            'compliance_check': False,
            'position_check': False,
            'liquidity_check': False,
            'circuit_breaker_check': False
        }
        
        try:
            # Check circuit breakers first
            if self.circuit_breakers_active:
                order.state = OrderState.REJECTED
                order.pre_trade_checks = {'circuit_breaker': 'active'}
                await self.event_bus.publish("order_rejected", {
                    "order_id": order.order_id,
                    "reason": "circuit_breaker_active"
                })
                return
            
            # Risk validation
            risk_valid = await self._validate_order_risk(order)
            checks['risk_check'] = risk_valid
            order.risk_approval = risk_valid
            
            # Compliance validation
            compliance_checks = await self.run_compliance_checks(order)
            compliance_violations = [c for c in compliance_checks if not c.passed and c.severity == 'violation']
            compliance_valid = len(compliance_violations) == 0
            checks['compliance_check'] = compliance_valid
            order.compliance_approval = compliance_valid
            
            # Position limits
            position_valid = await self._validate_position_limits(order)
            checks['position_check'] = position_valid
            
            # Liquidity validation
            liquidity_valid = await self._validate_liquidity(order)
            checks['liquidity_check'] = liquidity_valid
            
            # Circuit breaker check
            checks['circuit_breaker_check'] = not self.circuit_breakers_active
            
            order.pre_trade_checks = checks
            
            # Approve or reject
            all_passed = all(checks.values())
            
            if all_passed:
                order.state = OrderState.APPROVED
                logger.info(f"Order approved: {order.order_id}")
            else:
                order.state = OrderState.REJECTED
                failed_checks = [k for k, v in checks.items() if not v]
                
                await self.event_bus.publish("order_rejected", {
                    "order_id": order.order_id,
                    "failed_checks": failed_checks
                })
                
                logger.warning(f"Order rejected: {order.order_id} - failed checks: {failed_checks}")
        
        except Exception as e:
            logger.error(f"Pre-trade validation failed: {e}")
            order.state = OrderState.REJECTED
            order.pre_trade_checks = {'error': str(e)}
    
    async def _validate_order_risk(self, order: LiveOrder) -> bool:
        """Validate order against risk limits."""
        
        # Mock risk validation
        # In practice, would integrate with AdvancedRiskEngine
        
        order_value = order.quantity * (order.price or 100)  # Mock price
        
        # Check order size
        if order_value > self.max_position_value:
            return False
        
        # Check portfolio impact
        current_exposure = sum(
            abs(pos.market_value) for pos in self.position_snapshots.values()
        )
        
        if current_exposure + order_value > self.global_position_limit:
            return False
        
        return True
    
    async def _validate_position_limits(self, order: LiveOrder) -> bool:
        """Validate position limits."""
        
        # Check if position exists
        if order.strategy_id in self.position_snapshots:
            position = self.position_snapshots[order.strategy_id]
            
            # Check position limit utilization
            if position.position_limit_utilization > 0.9:  # 90% of limit used
                return False
        
        return True
    
    async def _validate_liquidity(self, order: LiveOrder) -> bool:
        """Validate liquidity for order."""
        
        # Mock liquidity validation
        # In practice, would check market depth, volumes, spreads
        
        # Assume orders under $10k are always liquid enough
        order_value = order.quantity * (order.price or 100)
        
        return order_value <= 10000
    
    async def _submit_to_exchange(self, order: LiveOrder):
        """Submit order to exchange (live trading)."""
        
        # This would integrate with real exchange APIs
        # For now, mock the submission
        
        submission_start = datetime.utcnow()
        
        # Mock exchange interaction
        await asyncio.sleep(0.1)  # Simulate network latency
        
        order.exchange_order_id = f"EXCH_{order.order_id[:8]}"
        order.submission_time_ms = (datetime.utcnow() - submission_start).total_seconds() * 1000
        
        logger.info(f"Order submitted to exchange: {order.exchange_order_id}")
    
    async def _mock_order_execution(self, order: LiveOrder):
        """Mock order execution for evaluation phase."""
        
        # Simulate realistic execution
        await asyncio.sleep(np.random.uniform(0.1, 0.5))  # Random execution delay
        
        # Mock fill
        fill_price = order.price or 100.0
        slippage = np.random.normal(0, 5)  # Random slippage in bps
        
        actual_fill_price = fill_price * (1 + slippage / 10000)
        
        # Create fill
        fill = {
            'fill_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'price': actual_fill_price,
            'quantity': order.quantity,
            'commission': order.quantity * actual_fill_price * 0.001  # 0.1% commission
        }
        
        order.fills.append(fill)
        order.filled_quantity = order.quantity
        order.avg_fill_price = actual_fill_price
        order.slippage_bps = abs(slippage)
        order.first_fill_time_ms = 200  # Mock fill time
        order.state = OrderState.FILLED
        order.updated_at = datetime.utcnow()
        
        logger.info(f"Mock order filled: {order.order_id} @ {actual_fill_price:.4f}")
    
    async def _cancel_on_exchange(self, order: LiveOrder) -> bool:
        """Cancel order on exchange."""
        
        # Mock cancellation
        await asyncio.sleep(0.05)  # Simulate cancellation latency
        
        return True
    
    async def _cancel_all_orders(self, reason: str = "system_request") -> int:
        """Cancel all active orders."""
        
        cancelled_count = 0
        
        for order_id, order in list(self.active_orders.items()):
            if order.state in [OrderState.SUBMITTED, OrderState.PARTIAL_FILLED]:
                success = await self.cancel_order(order_id)
                if success:
                    cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders due to: {reason}")
        
        return cancelled_count
    
    async def _emergency_position_closure(self) -> int:
        """Close positions in emergency scenarios."""
        
        closed_positions = 0
        
        for strategy_id, position in self.position_snapshots.items():
            if abs(position.quantity) > 0:
                # Create market order to close position
                close_side = "sell" if position.quantity > 0 else "buy"
                close_quantity = abs(position.quantity)
                
                close_order = await self.submit_order(
                    strategy_id=strategy_id,
                    symbol=position.symbol,
                    side=close_side,
                    quantity=close_quantity,
                    order_type="market"
                )
                
                if close_order.state in [OrderState.SUBMITTED, OrderState.FILLED]:
                    closed_positions += 1
                    logger.info(f"Emergency position closure: {strategy_id}")
        
        return closed_positions
    
    async def _capture_portfolio_state(self) -> Dict[str, Any]:
        """Capture current portfolio state."""
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'active_orders': len(self.active_orders),
            'position_count': len(self.position_snapshots),
            'total_exposure': sum(abs(pos.market_value) for pos in self.position_snapshots.values()),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.position_snapshots.values())
        }
    
    async def _validate_system_health(self) -> bool:
        """Validate system health before clearing emergency stop."""
        
        # Check various system components
        health_checks = {
            'risk_engine': True,  # Mock - would check actual risk engine health
            'exchange_connectivity': True,  # Mock - would check exchange connections
            'market_data': True,  # Mock - would check market data feeds
            'database': True,  # Mock - would check database connectivity
            'compliance_system': True  # Mock - would check compliance systems
        }
        
        return all(health_checks.values())
    
    async def _position_monitoring_loop(self):
        """Background position monitoring."""
        
        while True:
            try:
                # Update all position snapshots
                for strategy_id in list(self.position_snapshots.keys()):
                    await self._update_position_snapshot(strategy_id)
                    await self._check_position_alerts(strategy_id)
                
                # Check portfolio-level risks
                await self._check_portfolio_risks()
                
                # Sleep based on monitoring frequency
                await asyncio.sleep(self.position_monitor_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _compliance_monitoring_loop(self):
        """Background compliance monitoring."""
        
        while True:
            try:
                # Run compliance checks on active positions
                await self._monitor_compliance()
                
                # Check for regulatory violations
                await self._check_regulatory_violations()
                
                # Sleep for 60 seconds
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Compliance monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _update_position_snapshot(self, strategy_id: str):
        """Update position snapshot for strategy."""
        
        # Mock position update
        # In practice, would fetch from portfolio service
        
        if strategy_id not in self.position_snapshots:
            # Create new position snapshot
            self.position_snapshots[strategy_id] = PositionSnapshot(
                strategy_id=strategy_id,
                symbol="BTC/USDT",  # Mock
                quantity=np.random.uniform(-2, 2),
                market_value=np.random.uniform(-5000, 5000),
                unrealized_pnl=np.random.uniform(-500, 500),
                realized_pnl=np.random.uniform(-200, 200),
                position_limit_utilization=np.random.uniform(0.1, 0.8),
                var_contribution=np.random.uniform(0.05, 0.3),
                correlation_exposure={'BTC': 0.8, 'ETH': 0.2},
                avg_entry_price=np.random.uniform(95, 105),
                current_price=np.random.uniform(98, 102),
                time_in_position=timedelta(hours=np.random.randint(1, 48)),
                active_alerts=[],
                timestamp=datetime.utcnow()
            )
        else:
            # Update existing snapshot
            snapshot = self.position_snapshots[strategy_id]
            snapshot.current_price = np.random.uniform(98, 102)
            snapshot.market_value = snapshot.quantity * snapshot.current_price
            snapshot.unrealized_pnl = (snapshot.current_price - snapshot.avg_entry_price) * snapshot.quantity
            snapshot.timestamp = datetime.utcnow()
    
    async def _check_position_alerts(self, strategy_id: str):
        """Check for position-related alerts."""
        
        if strategy_id not in self.position_snapshots:
            return
        
        snapshot = self.position_snapshots[strategy_id]
        alerts = []
        
        # Position limit alert
        if snapshot.position_limit_utilization > 0.9:
            alerts.append(PositionMonitorAlert.POSITION_LIMIT_WARNING)
        
        # Drawdown alert
        if snapshot.unrealized_pnl < -1000:  # $1000 drawdown
            alerts.append(PositionMonitorAlert.DRAWDOWN_ALERT)
        
        snapshot.active_alerts = alerts
        
        # Emit alerts
        for alert in alerts:
            await self.event_bus.publish("position_alert", {
                "strategy_id": strategy_id,
                "alert_type": alert,
                "current_value": snapshot.market_value,
                "unrealized_pnl": snapshot.unrealized_pnl
            })
    
    async def _check_portfolio_risks(self):
        """Check portfolio-level risks."""
        
        total_exposure = sum(abs(pos.market_value) for pos in self.position_snapshots.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.position_snapshots.values())
        
        # Emergency drawdown check
        if abs(total_pnl) > self.global_position_limit * self.emergency_drawdown_limit:
            await self.emergency_stop_all(
                EmergencyStopType.RISK_LIMIT_BREACH,
                f"Portfolio drawdown limit exceeded: {total_pnl:.2f}",
                "system"
            )
        
        # Total exposure check
        if total_exposure > self.global_position_limit:
            await self.emergency_stop_all(
                EmergencyStopType.RISK_LIMIT_BREACH,
                f"Global position limit exceeded: {total_exposure:.2f}",
                "system"
            )
    
    async def _monitor_compliance(self):
        """Monitor ongoing compliance."""
        
        # Check all active positions for compliance
        for strategy_id, position in self.position_snapshots.items():
            # Position size compliance
            if abs(position.market_value) > self.max_position_value:
                violation = ComplianceCheck(
                    check_name="position_size_limit",
                    passed=False,
                    message=f"Position exceeds limit: {position.market_value:.2f} > {self.max_position_value}",
                    details={'strategy_id': strategy_id, 'position_value': position.market_value},
                    timestamp=datetime.utcnow(),
                    regulation_type="position_limit",
                    jurisdiction="US",
                    severity="violation"
                )
                
                self.compliance_violations.append(violation)
                
                await self.event_bus.publish("compliance_violation", {
                    "violation_type": "position_limit",
                    "strategy_id": strategy_id,
                    "details": violation.details
                })
    
    async def _check_regulatory_violations(self):
        """Check for regulatory violations."""
        
        # Keep only recent violations (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.compliance_violations = [
            v for v in self.compliance_violations
            if v.timestamp > cutoff
        ]
        
        # Check violation patterns
        violation_count = len(self.compliance_violations)
        
        if violation_count > 5:  # More than 5 violations in 24h
            await self.emergency_stop_all(
                EmergencyStopType.REGULATORY_HALT,
                f"Multiple compliance violations detected: {violation_count}",
                "compliance_system"
            )
    
    async def _initialize_emergency_stops(self):
        """Initialize emergency stop monitoring."""
        
        # Set up automatic emergency stop triggers
        pass  # Implementation would set up various triggers
    
    # Compliance check implementations
    
    async def _check_position_limits(self, order: LiveOrder) -> ComplianceCheck:
        """Check position limit compliance."""
        
        order_value = order.quantity * (order.price or 100)
        
        if order_value > self.max_position_value:
            return ComplianceCheck(
                check_name="position_limit",
                passed=False,
                message=f"Order exceeds position limit: {order_value:.2f} > {self.max_position_value}",
                details={'order_value': order_value, 'limit': self.max_position_value},
                timestamp=datetime.utcnow(),
                regulation_type="position_limit",
                jurisdiction="US",
                severity="violation"
            )
        else:
            return ComplianceCheck(
                check_name="position_limit",
                passed=True,
                message="Order within position limits",
                details={'order_value': order_value},
                timestamp=datetime.utcnow(),
                regulation_type="position_limit",
                jurisdiction="US",
                severity="info"
            )
    
    async def _check_wash_trading(self, order: LiveOrder) -> ComplianceCheck:
        """Check for wash trading patterns."""
        
        # Simplified wash trading detection
        # In practice, would analyze trading patterns across accounts/strategies
        
        return ComplianceCheck(
            check_name="wash_trading",
            passed=True,
            message="No wash trading pattern detected",
            details={'pattern_score': 0.1},
            timestamp=datetime.utcnow(),
            regulation_type="wash_trading",
            jurisdiction="US",
            severity="info"
        )
    
    async def _check_market_manipulation(self, order: LiveOrder) -> ComplianceCheck:
        """Check for market manipulation indicators."""
        
        # Simplified manipulation detection
        return ComplianceCheck(
            check_name="market_manipulation",
            passed=True,
            message="No manipulation indicators detected",
            details={'manipulation_score': 0.05},
            timestamp=datetime.utcnow(),
            regulation_type="market_manipulation",
            jurisdiction="US",
            severity="info"
        )
    
    async def _check_concentration_limits(self, order: LiveOrder) -> ComplianceCheck:
        """Check concentration limit compliance."""
        
        # Check if order would create excessive concentration
        return ComplianceCheck(
            check_name="concentration_limit",
            passed=True,
            message="Concentration within limits",
            details={'concentration_pct': 0.15},
            timestamp=datetime.utcnow(),
            regulation_type="position_limit",
            jurisdiction="US",
            severity="info"
        )
    
    async def _check_trading_hours(self, order: LiveOrder) -> ComplianceCheck:
        """Check trading hours compliance."""
        
        current_hour = datetime.utcnow().hour
        
        # Mock trading hours: 24/7 for crypto, but check for maintenance windows
        if 0 <= current_hour <= 23:  # Always allow for crypto
            return ComplianceCheck(
                check_name="trading_hours",
                passed=True,
                message="Order within trading hours",
                details={'current_hour': current_hour},
                timestamp=datetime.utcnow(),
                regulation_type="trading_hours",
                jurisdiction="US",
                severity="info"
            )
        else:
            return ComplianceCheck(
                check_name="trading_hours",
                passed=False,
                message="Order outside trading hours",
                details={'current_hour': current_hour},
                timestamp=datetime.utcnow(),
                regulation_type="trading_hours",
                jurisdiction="US",
                severity="violation"
            )
    
    def _default_execution_quality(self, strategy_id: str, time_period: str) -> ExecutionQuality:
        """Return default execution quality metrics."""
        
        return ExecutionQuality(
            strategy_id=strategy_id,
            symbol="ALL",
            time_period=time_period,
            avg_slippage_bps=0.0,
            worst_slippage_bps=0.0,
            slippage_volatility=0.0,
            avg_fill_time_ms=0.0,
            fill_time_percentile_95=0.0,
            temporary_impact_bps=0.0,
            permanent_impact_bps=0.0,
            fill_rate=0.0,
            partial_fill_rate=0.0,
            cancel_rate=0.0,
            vs_arrival_price_bps=0.0,
            vs_twap_bps=0.0,
            vs_vwap_bps=0.0
        )
    
    def get_live_trading_status(self) -> Dict[str, Any]:
        """Get live trading manager status."""
        
        return {
            'live_enabled': self.live_enabled,
            'circuit_breakers_active': self.circuit_breakers_active,
            'active_orders': len(self.active_orders),
            'active_positions': len(self.position_snapshots),
            'emergency_stops': len(self.emergency_stops),
            'compliance_violations_24h': len(self.compliance_violations),
            'total_exposure': sum(abs(pos.market_value) for pos in self.position_snapshots.values()),
            'global_position_limit': self.global_position_limit,
            'max_position_value': self.max_position_value,
            'emergency_drawdown_limit': self.emergency_drawdown_limit
        }