"""
SMA Crossover Strategy
Simple Moving Average crossover strategy with trend following logic.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..services.strategy_engine import BaseStrategy, TradingSignal, SignalType
from ..services.market_data import OHLCV

logger = logging.getLogger(__name__)

class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        super().__init__(strategy_id, config)
        
        # Strategy parameters
        self.fast_period = config.get('fast_period', 20)
        self.slow_period = config.get('slow_period', 50)
        self.min_trend_strength = config.get('min_trend_strength', 0.001)  # 0.1% minimum trend
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4% take profit
        self.volume_filter = config.get('volume_filter', True)
        self.min_volume_ratio = config.get('min_volume_ratio', 1.2)  # 20% above average
        
        # State tracking
        self.last_crossover: Dict[str, Dict] = {}  # Track last crossover for each symbol
        self.entry_prices: Dict[str, float] = {}  # Track entry prices for stop/take profit
    
    def get_required_indicators(self) -> List[Dict[str, Any]]:
        """Get required indicators for this strategy."""
        return [
            {'name': 'SMA', 'params': {'period': self.fast_period}},
            {'name': 'SMA', 'params': {'period': self.slow_period}},
            {'name': 'ATR', 'params': {'period': 14}},  # For volatility assessment
        ]
    
    async def analyze(self, 
                     symbol: str, 
                     market_data: List[OHLCV], 
                     indicators: Dict[str, Any]) -> Optional[TradingSignal]:
        """Analyze market data and generate trading signals."""
        
        try:
            if len(market_data) < max(self.fast_period, self.slow_period) + 10:
                logger.debug(f"Insufficient data for {symbol}: need {max(self.fast_period, self.slow_period) + 10}, got {len(market_data)}")
                return None
            
            # Get current and previous prices
            current_price = market_data[-1].close
            previous_price = market_data[-2].close if len(market_data) > 1 else current_price
            current_volume = market_data[-1].volume
            
            # Get SMA values
            fast_sma_values = indicators.get('SMA', {})
            if not hasattr(fast_sma_values, 'values') or len(fast_sma_values.values) < 2:
                logger.debug(f"No fast SMA data for {symbol}")
                return None
            
            # We need to get both SMAs - the indicator engine should return both
            # For now, we'll calculate them separately
            closes = [candle.close for candle in market_data]
            
            # Calculate SMAs manually if needed
            fast_sma_current = sum(closes[-self.fast_period:]) / self.fast_period
            fast_sma_previous = sum(closes[-self.fast_period-1:-1]) / self.fast_period
            
            slow_sma_current = sum(closes[-self.slow_period:]) / self.slow_period
            slow_sma_previous = sum(closes[-self.slow_period-1:-1]) / self.slow_period
            
            # Calculate average volume for volume filter
            volumes = [candle.volume for candle in market_data[-20:]]  # Last 20 periods
            avg_volume = sum(volumes) / len(volumes)
            
            # Get ATR for volatility assessment
            atr_value = 0.0
            if 'ATR' in indicators and hasattr(indicators['ATR'], 'values'):
                atr_values = indicators['ATR'].values
                if atr_values and len(atr_values) > 0:
                    atr_value = atr_values[-1]
            
            # Detect crossovers
            bullish_crossover = (fast_sma_current > slow_sma_current and 
                               fast_sma_previous <= slow_sma_previous)
            
            bearish_crossover = (fast_sma_current < slow_sma_current and 
                               fast_sma_previous >= slow_sma_previous)
            
            # Volume filter
            volume_condition = True
            if self.volume_filter:
                volume_condition = current_volume >= (avg_volume * self.min_volume_ratio)
            
            # Trend strength filter
            trend_strength = abs(fast_sma_current - slow_sma_current) / current_price
            strong_trend = trend_strength >= self.min_trend_strength
            
            # Volatility filter - avoid trading in extremely volatile conditions
            volatility_ok = True
            if atr_value > 0:
                volatility_pct = atr_value / current_price
                volatility_ok = volatility_pct < 0.05  # Less than 5% volatility
            
            # Check for entry signals
            if bullish_crossover and strong_trend and volume_condition and volatility_ok:
                # Calculate stop loss and take profit
                stop_loss_price = current_price * (1 - self.stop_loss_pct)
                take_profit_price = current_price * (1 + self.take_profit_pct)
                
                # Calculate position size based on ATR (risk-based sizing)
                if atr_value > 0:
                    risk_per_share = max(atr_value * 2, current_price * self.stop_loss_pct)
                    # This would be adjusted based on portfolio size and risk tolerance
                    quantity = None  # Let the strategy engine calculate default size
                else:
                    quantity = None
                
                # Store entry price for future reference
                self.entry_prices[symbol] = current_price
                
                # Store crossover info
                self.last_crossover[symbol] = {
                    'type': 'bullish',
                    'timestamp': datetime.utcnow(),
                    'fast_sma': fast_sma_current,
                    'slow_sma': slow_sma_current,
                    'price': current_price,
                    'volume_ratio': current_volume / avg_volume,
                    'trend_strength': trend_strength
                }
                
                return TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(trend_strength * 10, 1.0),  # Scale to 0-1
                    price=current_price,
                    quantity=quantity,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    metadata={
                        'fast_sma': fast_sma_current,
                        'slow_sma': slow_sma_current,
                        'trend_strength': trend_strength,
                        'volume_ratio': current_volume / avg_volume,
                        'atr': atr_value,
                        'crossover_type': 'bullish',
                        'entry_reason': 'SMA bullish crossover with strong trend'
                    }
                )
            
            elif bearish_crossover and strong_trend and volume_condition and volatility_ok:
                # Check if we have a long position to close
                entry_price = self.entry_prices.get(symbol)
                if entry_price:
                    # Calculate P&L
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Store crossover info
                    self.last_crossover[symbol] = {
                        'type': 'bearish',
                        'timestamp': datetime.utcnow(),
                        'fast_sma': fast_sma_current,
                        'slow_sma': slow_sma_current,
                        'price': current_price,
                        'volume_ratio': current_volume / avg_volume,
                        'trend_strength': trend_strength,
                        'exit_pnl_pct': pnl_pct
                    }
                    
                    # Clear entry price
                    del self.entry_prices[symbol]
                    
                    return TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=min(trend_strength * 10, 1.0),
                        price=current_price,
                        metadata={
                            'fast_sma': fast_sma_current,
                            'slow_sma': slow_sma_current,
                            'trend_strength': trend_strength,
                            'volume_ratio': current_volume / avg_volume,
                            'atr': atr_value,
                            'crossover_type': 'bearish',
                            'exit_reason': 'SMA bearish crossover',
                            'entry_price': entry_price,
                            'exit_pnl_pct': pnl_pct
                        }
                    )
            
            # Check for stop loss or take profit if we have a position
            entry_price = self.entry_prices.get(symbol)
            if entry_price:
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Stop loss check
                if pnl_pct <= -self.stop_loss_pct:
                    del self.entry_prices[symbol]
                    
                    return TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=1.0,  # High strength for stop loss
                        price=current_price,
                        metadata={
                            'exit_reason': 'Stop loss triggered',
                            'entry_price': entry_price,
                            'exit_pnl_pct': pnl_pct,
                            'stop_loss_pct': self.stop_loss_pct
                        }
                    )
                
                # Take profit check
                elif pnl_pct >= self.take_profit_pct:
                    del self.entry_prices[symbol]
                    
                    return TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=0.8,  # High strength for take profit
                        price=current_price,
                        metadata={
                            'exit_reason': 'Take profit triggered',
                            'entry_price': entry_price,
                            'exit_pnl_pct': pnl_pct,
                            'take_profit_pct': self.take_profit_pct
                        }
                    )
            
            # No signal
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strength=0.0,
                price=current_price,
                metadata={
                    'fast_sma': fast_sma_current,
                    'slow_sma': slow_sma_current,
                    'trend_strength': trend_strength,
                    'volume_condition': volume_condition,
                    'volatility_ok': volatility_ok,
                    'strong_trend': strong_trend,
                    'reason': 'No crossover or insufficient conditions'
                }
            )
            
        except Exception as e:
            logger.error(f"Error in SMA crossover analysis for {symbol}: {e}")
            return None
    
    def validate_config(self) -> List[str]:
        """Validate strategy configuration."""
        errors = super().validate_config()
        
        if self.fast_period >= self.slow_period:
            errors.append("Fast period must be less than slow period")
        
        if self.fast_period < 5:
            errors.append("Fast period must be at least 5")
        
        if self.slow_period < 10:
            errors.append("Slow period must be at least 10")
        
        if not (0 < self.stop_loss_pct < 0.1):
            errors.append("Stop loss percentage must be between 0% and 10%")
        
        if not (0 < self.take_profit_pct < 0.2):
            errors.append("Take profit percentage must be between 0% and 20%")
        
        if self.take_profit_pct <= self.stop_loss_pct:
            errors.append("Take profit must be greater than stop loss")
        
        return errors
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state."""
        base_info = super().get_status()
        
        return {
            **base_info,
            'strategy_type': 'SMA Crossover',
            'description': f'Simple Moving Average crossover strategy ({self.fast_period}/{self.slow_period})',
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'min_trend_strength': self.min_trend_strength,
                'volume_filter': self.volume_filter,
                'min_volume_ratio': self.min_volume_ratio
            },
            'current_positions': list(self.entry_prices.keys()),
            'last_crossovers': {
                symbol: {
                    **crossover,
                    'timestamp': crossover['timestamp'].isoformat()
                }
                for symbol, crossover in self.last_crossover.items()
            },
            'active_entries': len(self.entry_prices)
        }

def create_sma_crossover_strategy(strategy_id: str, config: Dict[str, Any]) -> SMACrossoverStrategy:
    """Factory function to create SMA Crossover strategy."""
    
    # Default configuration
    default_config = {
        'name': 'SMA Crossover',
        'symbols': ['BTC/USDT'],
        'timeframe': '1h',
        'fast_period': 20,
        'slow_period': 50,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'min_trend_strength': 0.001,
        'volume_filter': True,
        'min_volume_ratio': 1.2
    }
    
    # Merge with provided config
    merged_config = {**default_config, **config}
    
    return SMACrossoverStrategy(strategy_id, merged_config)