"""
Comprehensive Risk Engine
Position limits, daily loss limits, circuit breakers, and risk monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from decimal import Decimal

from ..config.settings import get_settings
from ..services.event_bus import EventBus
from ..models.trading import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskViolationType(str, Enum):
    POSITION_SIZE = "position_size"
    DAILY_LOSS = "daily_loss"
    ORDER_FREQUENCY = "order_frequency"
    SPREAD_WARNING = "spread_warning"
    CIRCUIT_BREAKER = "circuit_breaker"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    CORRELATION_LIMIT = "correlation_limit"

@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_pct: float = 0.25  # Max position as % of portfolio
    daily_loss_limit_pct: float = 2.0  # Max daily loss as % of portfolio
    max_orders_per_hour: int = 6  # Max orders per hour
    circuit_breaker_spread_bps: int = 50  # Circuit breaker spread threshold
    max_correlation_exposure: float = 0.5  # Max exposure to correlated assets
    max_single_symbol_pct: float = 0.3  # Max single symbol exposure
    stop_loss_threshold_pct: float = 10.0  # Auto stop loss threshold
    drawdown_limit_pct: float = 15.0  # Max drawdown from peak
    margin_buffer_pct: float = 0.2  # Margin buffer requirement

@dataclass
class RiskViolation:
    """Risk violation record."""
    violation_type: RiskViolationType
    severity: RiskLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    symbol: Optional[str] = None
    order_id: Optional[str] = None

@dataclass
class RiskMetrics:
    """Current risk metrics."""
    portfolio_value: float
    total_exposure: float
    exposure_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    var_1day: float  # Value at Risk (1 day)
    sharpe_ratio: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    correlation_exposure: float
    risk_score: float  # 0-100
    risk_level: RiskLevel
    last_updated: datetime

class RiskEngine:
    """Comprehensive risk management engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.settings = get_settings()
        
        # Risk limits
        self.limits = RiskLimits(
            max_position_pct=self.settings.risk.max_position_pct,
            daily_loss_limit_pct=self.settings.risk.daily_loss_limit_pct,
            max_orders_per_hour=self.settings.risk.max_orders_per_hour,
            circuit_breaker_spread_bps=self.settings.risk.circuit_breaker_spread_bps
        )
        
        # Risk monitoring state
        self.violations: List[RiskViolation] = []
        self.order_history: List[Tuple[datetime, str, str]] = []  # (timestamp, symbol, order_id)
        self.daily_pnl_history: Dict[str, float] = {}  # date -> pnl
        self.portfolio_peaks: Dict[str, float] = {}  # date -> peak value
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_until: Optional[datetime] = None
        
        # Kill switch
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
        
        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start risk engine."""
        logger.info("Starting risk engine...")
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Risk engine started")
    
    async def stop(self):
        """Stop risk engine."""
        logger.info("Stopping risk engine...")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk engine stopped")
    
    async def validate_order(self, 
                            order: Order, 
                            current_positions: Dict[str, Dict],
                            portfolio_value: float,
                            current_price: float) -> Tuple[bool, List[RiskViolation]]:
        """Validate an order against risk limits."""
        violations = []
        
        # Check kill switch
        if self.kill_switch_active:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.CIRCUIT_BREAKER,
                severity=RiskLevel.CRITICAL,
                message=f"Kill switch active: {self.kill_switch_reason}",
                details={'kill_switch': True},
                timestamp=datetime.utcnow(),
                symbol=order.symbol,
                order_id=order.id
            ))
            return False, violations
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            if datetime.utcnow() < self.circuit_breaker_until:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.CIRCUIT_BREAKER,
                    severity=RiskLevel.HIGH,
                    message="Circuit breaker active",
                    details={'until': self.circuit_breaker_until.isoformat()},
                    timestamp=datetime.utcnow(),
                    symbol=order.symbol,
                    order_id=order.id
                ))
                return False, violations
            else:
                # Circuit breaker expired
                await self.deactivate_circuit_breaker()
        
        # Check order frequency
        frequency_violation = await self._check_order_frequency(order.symbol)
        if frequency_violation:
            violations.append(frequency_violation)
        
        # Check position size limits
        position_violations = await self._check_position_limits(
            order, current_positions, portfolio_value, current_price
        )
        violations.extend(position_violations)
        
        # Check daily loss limits
        daily_loss_violation = await self._check_daily_loss_limit(portfolio_value)
        if daily_loss_violation:
            violations.append(daily_loss_violation)
        
        # Check balance sufficiency
        balance_violation = await self._check_sufficient_balance(
            order, portfolio_value, current_price
        )
        if balance_violation:
            violations.append(balance_violation)
        
        # Check spread conditions
        spread_violation = await self._check_spread_conditions(order.symbol, current_price)
        if spread_violation:
            violations.append(spread_violation)
        
        # Determine if order should be approved
        critical_violations = [v for v in violations if v.severity == RiskLevel.CRITICAL]
        high_violations = [v for v in violations if v.severity == RiskLevel.HIGH]
        
        # Reject if critical violations or too many high violations
        should_approve = len(critical_violations) == 0 and len(high_violations) <= 1
        
        # Log violations
        for violation in violations:
            await self._log_violation(violation)
        
        return should_approve, violations
    
    async def _check_order_frequency(self, symbol: str) -> Optional[RiskViolation]:
        """Check order frequency limits."""
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        
        # Count recent orders
        recent_orders = [
            order for order in self.order_history 
            if order[0] > one_hour_ago and order[1] == symbol
        ]
        
        if len(recent_orders) >= self.limits.max_orders_per_hour:
            return RiskViolation(
                violation_type=RiskViolationType.ORDER_FREQUENCY,
                severity=RiskLevel.HIGH,
                message=f"Order frequency limit exceeded: {len(recent_orders)}/{self.limits.max_orders_per_hour}",
                details={
                    'recent_orders': len(recent_orders),
                    'limit': self.limits.max_orders_per_hour,
                    'symbol': symbol
                },
                timestamp=now,
                symbol=symbol
            )
        
        return None
    
    async def _check_position_limits(self, 
                                   order: Order,
                                   current_positions: Dict[str, Dict],
                                   portfolio_value: float,
                                   current_price: float) -> List[RiskViolation]:
        """Check position size limits."""
        violations = []
        
        # Calculate order value
        order_value = float(order.qty) * current_price
        
        # Check single position limit
        current_position = current_positions.get(order.symbol, {})
        current_exposure = abs(current_position.get('market_value', 0.0))
        
        if order.side == OrderSide.BUY:
            new_exposure = current_exposure + order_value
        else:
            # For sells, exposure might decrease or increase (short)
            current_qty = current_position.get('quantity', 0.0)
            if current_qty > 0:
                # Reducing long position
                new_exposure = max(0, current_exposure - order_value)
            else:
                # Adding to short or creating short
                new_exposure = current_exposure + order_value
        
        position_pct = new_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if position_pct > self.limits.max_position_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE,
                severity=RiskLevel.HIGH,
                message=f"Position size limit exceeded: {position_pct:.2%} > {self.limits.max_position_pct:.2%}",
                details={
                    'position_pct': position_pct,
                    'limit_pct': self.limits.max_position_pct,
                    'order_value': order_value,
                    'current_exposure': current_exposure
                },
                timestamp=datetime.utcnow(),
                symbol=order.symbol,
                order_id=order.id
            ))
        
        # Check total portfolio exposure
        total_exposure = sum(abs(pos.get('market_value', 0.0)) for pos in current_positions.values())
        total_exposure += order_value if order.side == OrderSide.BUY else 0
        
        total_exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if total_exposure_pct > 1.0:  # 100% exposure limit
            violations.append(RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE,
                severity=RiskLevel.MEDIUM,
                message=f"Total exposure limit exceeded: {total_exposure_pct:.2%}",
                details={
                    'total_exposure_pct': total_exposure_pct,
                    'limit_pct': 1.0
                },
                timestamp=datetime.utcnow(),
                symbol=order.symbol,
                order_id=order.id
            ))
        
        return violations
    
    async def _check_daily_loss_limit(self, portfolio_value: float) -> Optional[RiskViolation]:
        """Check daily loss limits."""
        today = datetime.utcnow().date().isoformat()
        
        # Get today's PnL (this would come from portfolio service)
        daily_pnl = self.daily_pnl_history.get(today, 0.0)
        daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0
        
        if daily_pnl < 0 and daily_loss_pct > self.limits.daily_loss_limit_pct / 100:
            return RiskViolation(
                violation_type=RiskViolationType.DAILY_LOSS,
                severity=RiskLevel.CRITICAL,
                message=f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.limits.daily_loss_limit_pct:.1f}%",
                details={
                    'daily_pnl': daily_pnl,
                    'daily_loss_pct': daily_loss_pct,
                    'limit_pct': self.limits.daily_loss_limit_pct / 100
                },
                timestamp=datetime.utcnow()
            )
        
        return None
    
    async def _check_sufficient_balance(self, 
                                      order: Order, 
                                      portfolio_value: float,
                                      current_price: float) -> Optional[RiskViolation]:
        """Check if there's sufficient balance for the order."""
        order_value = float(order.qty) * current_price
        
        # Add margin buffer
        required_balance = order_value * (1 + self.limits.margin_buffer_pct)
        
        if order.side == OrderSide.BUY and required_balance > portfolio_value:
            return RiskViolation(
                violation_type=RiskViolationType.INSUFFICIENT_BALANCE,
                severity=RiskLevel.HIGH,
                message=f"Insufficient balance: need {required_balance:.2f}, have {portfolio_value:.2f}",
                details={
                    'required_balance': required_balance,
                    'available_balance': portfolio_value,
                    'order_value': order_value,
                    'margin_buffer': self.limits.margin_buffer_pct
                },
                timestamp=datetime.utcnow(),
                symbol=order.symbol,
                order_id=order.id
            )
        
        return None
    
    async def _check_spread_conditions(self, 
                                     symbol: str, 
                                     current_price: float) -> Optional[RiskViolation]:
        """Check market spread conditions."""
        # This would integrate with market data service
        # For now, simulate spread checking
        
        # Mock spread calculation (would be real in production)
        mock_spread_bps = 10  # 0.1% spread
        
        if mock_spread_bps > self.limits.circuit_breaker_spread_bps:
            return RiskViolation(
                violation_type=RiskViolationType.SPREAD_WARNING,
                severity=RiskLevel.MEDIUM,
                message=f"Wide spread detected: {mock_spread_bps} bps > {self.limits.circuit_breaker_spread_bps} bps",
                details={
                    'spread_bps': mock_spread_bps,
                    'limit_bps': self.limits.circuit_breaker_spread_bps,
                    'price': current_price
                },
                timestamp=datetime.utcnow(),
                symbol=symbol
            )
        
        return None
    
    async def update_portfolio_state(self, 
                                   portfolio_value: float,
                                   positions: Dict[str, Dict],
                                   daily_pnl: float):
        """Update portfolio state for risk monitoring."""
        today = datetime.utcnow().date().isoformat()
        
        # Update daily PnL
        self.daily_pnl_history[today] = daily_pnl
        
        # Update portfolio peaks for drawdown calculation
        current_peak = self.portfolio_peaks.get(today, portfolio_value)
        self.portfolio_peaks[today] = max(current_peak, portfolio_value)
        
        # Check for automatic risk actions
        await self._check_automatic_actions(portfolio_value, positions, daily_pnl)
    
    async def _check_automatic_actions(self, 
                                     portfolio_value: float,
                                     positions: Dict[str, Dict],
                                     daily_pnl: float):
        """Check if automatic risk actions should be triggered."""
        
        # Check for circuit breaker activation
        daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0
        
        if daily_pnl < 0 and daily_loss_pct > (self.limits.daily_loss_limit_pct / 100) * 0.8:
            # Approaching daily loss limit (80% of limit)
            await self.activate_circuit_breaker("Approaching daily loss limit", duration_minutes=30)
        
        # Check drawdown
        today = datetime.utcnow().date().isoformat()
        peak_value = self.portfolio_peaks.get(today, portfolio_value)
        drawdown_pct = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        
        if drawdown_pct > self.limits.drawdown_limit_pct / 100:
            await self.activate_kill_switch(f"Maximum drawdown exceeded: {drawdown_pct:.2%}")
    
    async def activate_circuit_breaker(self, reason: str, duration_minutes: int = 60):
        """Activate circuit breaker."""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        violation = RiskViolation(
            violation_type=RiskViolationType.CIRCUIT_BREAKER,
            severity=RiskLevel.HIGH,
            message=f"Circuit breaker activated: {reason}",
            details={
                'reason': reason,
                'until': self.circuit_breaker_until.isoformat(),
                'duration_minutes': duration_minutes
            },
            timestamp=datetime.utcnow()
        )
        
        await self._log_violation(violation)
        
        # Emit event
        await self.event_bus.publish("risk", {
            "type": "circuit_breaker_activated",
            "reason": reason,
            "until": self.circuit_breaker_until.isoformat()
        })
        
        logger.warning(f"Circuit breaker activated: {reason}")
    
    async def deactivate_circuit_breaker(self):
        """Deactivate circuit breaker."""
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        
        # Emit event
        await self.event_bus.publish("risk", {
            "type": "circuit_breaker_deactivated",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("Circuit breaker deactivated")
    
    async def activate_kill_switch(self, reason: str):
        """Activate kill switch - emergency stop all trading."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        
        violation = RiskViolation(
            violation_type=RiskViolationType.CIRCUIT_BREAKER,
            severity=RiskLevel.CRITICAL,
            message=f"Kill switch activated: {reason}",
            details={'reason': reason},
            timestamp=datetime.utcnow()
        )
        
        await self._log_violation(violation)
        
        # Emit event
        await self.event_bus.publish("risk", {
            "type": "kill_switch_activated",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    
    async def deactivate_kill_switch(self):
        """Deactivate kill switch."""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        
        # Emit event
        await self.event_bus.publish("risk", {
            "type": "kill_switch_deactivated",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("Kill switch deactivated")
    
    async def record_order(self, order: Order):
        """Record order for frequency tracking."""
        self.order_history.append((
            datetime.utcnow(),
            order.symbol,
            order.id
        ))
        
        # Keep only last 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.order_history = [
            entry for entry in self.order_history 
            if entry[0] > cutoff
        ]
    
    async def calculate_risk_metrics(self, 
                                   portfolio_value: float,
                                   positions: Dict[str, Dict],
                                   daily_pnl: float) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        
        # Calculate exposures
        total_exposure = sum(abs(pos.get('market_value', 0.0)) for pos in positions.values())
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate daily PnL percentage
        daily_pnl_pct = daily_pnl / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate drawdown
        today = datetime.utcnow().date().isoformat()
        peak_value = self.portfolio_peaks.get(today, portfolio_value)
        current_drawdown_pct = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        
        # Mock calculations for advanced metrics (would be real in production)
        var_1day = portfolio_value * 0.02  # 2% VaR
        sharpe_ratio = 1.5  # Mock Sharpe ratio
        max_drawdown_pct = 0.05  # Mock max drawdown
        correlation_exposure = 0.3  # Mock correlation exposure
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(
            exposure_pct, daily_pnl_pct, current_drawdown_pct
        )
        
        # Determine risk level
        if risk_score < 25:
            risk_level = RiskLevel.LOW
        elif risk_score < 50:
            risk_level = RiskLevel.MEDIUM
        elif risk_score < 75:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            var_1day=var_1day,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            current_drawdown_pct=current_drawdown_pct,
            correlation_exposure=correlation_exposure,
            risk_score=risk_score,
            risk_level=risk_level,
            last_updated=datetime.utcnow()
        )
    
    def _calculate_risk_score(self, 
                            exposure_pct: float, 
                            daily_pnl_pct: float, 
                            drawdown_pct: float) -> float:
        """Calculate overall risk score (0-100)."""
        
        # Exposure component (0-30 points)
        exposure_score = min(exposure_pct * 30, 30)
        
        # Daily loss component (0-40 points)
        daily_loss_score = 0
        if daily_pnl_pct < 0:
            daily_loss_score = min(abs(daily_pnl_pct) * 20, 40)
        
        # Drawdown component (0-30 points)
        drawdown_score = min(drawdown_pct * 20, 30)
        
        # Violation penalty
        recent_violations = [
            v for v in self.violations 
            if v.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        violation_penalty = min(len(recent_violations) * 5, 20)
        
        total_score = exposure_score + daily_loss_score + drawdown_score + violation_penalty
        return min(total_score, 100.0)
    
    async def _log_violation(self, violation: RiskViolation):
        """Log a risk violation."""
        self.violations.append(violation)
        
        # Keep only recent violations (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.violations = [v for v in self.violations if v.timestamp > cutoff]
        
        # Emit event
        await self.event_bus.publish("risk", {
            "type": "violation",
            "violation": asdict(violation)
        })
        
        logger.warning(f"Risk violation: {violation.message}")
    
    async def _monitoring_loop(self):
        """Background risk monitoring loop."""
        while True:
            try:
                # Clean up old data
                await self._cleanup_old_data()
                
                # Check for expired circuit breakers
                if (self.circuit_breaker_active and 
                    self.circuit_breaker_until and 
                    datetime.utcnow() > self.circuit_breaker_until):
                    await self.deactivate_circuit_breaker()
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        # Clean order history (keep 24 hours)
        cutoff_orders = datetime.utcnow() - timedelta(hours=24)
        self.order_history = [
            entry for entry in self.order_history 
            if entry[0] > cutoff_orders
        ]
        
        # Clean violations (keep 24 hours)
        cutoff_violations = datetime.utcnow() - timedelta(hours=24)
        self.violations = [
            v for v in self.violations 
            if v.timestamp > cutoff_violations
        ]
        
        # Clean old daily PnL data (keep 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        cutoff_date_str = cutoff_date.date().isoformat()
        
        self.daily_pnl_history = {
            date: pnl for date, pnl in self.daily_pnl_history.items()
            if date >= cutoff_date_str
        }
        
        self.portfolio_peaks = {
            date: peak for date, peak in self.portfolio_peaks.items()
            if date >= cutoff_date_str
        }
    
    def get_limits(self) -> RiskLimits:
        """Get current risk limits."""
        return self.limits
    
    def update_limits(self, new_limits: RiskLimits):
        """Update risk limits."""
        self.limits = new_limits
        logger.info("Risk limits updated")
    
    def get_recent_violations(self, hours: int = 24) -> List[RiskViolation]:
        """Get recent violations."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [v for v in self.violations if v.timestamp > cutoff]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk engine status."""
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_until': self.circuit_breaker_until.isoformat() if self.circuit_breaker_until else None,
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            'recent_violations': len(self.get_recent_violations(1)),  # Last hour
            'total_violations_24h': len(self.get_recent_violations(24)),
            'limits': asdict(self.limits)
        }